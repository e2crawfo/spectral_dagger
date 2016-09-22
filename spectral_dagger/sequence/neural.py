# Based on code from the Theano website.
from __future__ import print_function
import six
from collections import OrderedDict
import time
import numpy as np
from scipy.stats import multivariate_normal, bernoulli

import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_random_state

from spectral_dagger.sequence import SequenceModel
from spectral_dagger import gen_seed, Reals
from spectral_dagger.utils.cache import get_cache_key


def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """ Used to shuffle the dataset at each iteration. """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return list(enumerate(minibatches))


def unzip_weights(zipped, theano_weights):
    """ Extract parameter values stored in ``weights`` inside ``theano_weights``,
        a set of active shared Theano variables. """
    for kk, vv in zipped.items():
        theano_weights[kk].set_value(vv)


def zip_weights(unzipped):
    """ Store parameter values from ``theano_weights`` (an active set of shared
        Theano variables) inside ``weights``. """
    new_weights = OrderedDict()
    for kk, vv in unzipped.items():
        new_weights[kk] = vv.get_value()
    return new_weights


def _p(pp, name):
    return '%s_%s' % (pp, name)


def load_weights(path, weights):
    pp = np.load(path)
    for kk, vv in weights.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        weights[kk] = pp[kk]

    return weights


def as_shared_theano(d):
    """ Convert a dictionary which maps from (names to initial parameter values) to
        a dictionary which maps from (names to Theano Shared Variables). """
    shared = OrderedDict()
    for k, v in six.iteritems(d):
        shared[k] = theano.shared(v, name=k)
    return shared


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def init_W(n_input, n_hidden):
    W = ortho_weight(max(n_input, n_hidden))
    return W[:n_input, :n_hidden]


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(
        use_noise,
        (state_before * trng.binomial(
            state_before.shape, p=0.5, n=1, dtype=state_before.dtype)),
        state_before * 0.5)
    return proj


def rnn_layer(theano_weights, state_below, options, prefix='rnn', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _step(m_, x_, h_):
        preact = T.dot(h_, theano_weights['rnn_U'])
        preact += x_
        h = options['recurrent_nl'](_slice(preact, 0, n_hidden))
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    state_below = (T.dot(state_below, theano_weights['rnn_W']) +
                   theano_weights['rnn_b'])

    n_hidden = options['n_hidden']
    rval, updates = theano.scan(
        _step,
        sequences=[mask, state_below],
        outputs_info=T.alloc(np_floatX(0.), n_samples, n_hidden),
        name='rnn_layers',
        n_steps=nsteps)

    return rval


def gru_layer(theano_weights, state_below, options, prefix='gru', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _step(m_, x_, h_):
        n_hidden = options['n_hidden']
        preact = T.dot(h_, theano_weights['gru_U']) + _slice(x_, 0, 2*n_hidden)

        r = T.nnet.sigmoid(_slice(preact, 0, n_hidden))  # reset gate
        u = T.nnet.sigmoid(_slice(preact, 1, n_hidden))  # update gate

        h = T.tanh(T.dot(r*h_, theano_weights['gru_U_blank']) + _slice(x_, 2, n_hidden))
        h = (1. - u) * h_ + u * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    state_below = (T.dot(state_below, theano_weights[_p(prefix, 'W')]) +
                   theano_weights[_p(prefix, 'b')])

    n_hidden = options['n_hidden']
    rval, updates = theano.scan(
        _step,
        sequences=[mask, state_below],
        outputs_info=T.alloc(np_floatX(0.), n_samples, n_hidden),
        name=_p(prefix, '_layers'),
        n_steps=nsteps)

    return rval


def lstm_layer(theano_weights, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, theano_weights['lstm_U'])
        preact += x_

        n_hidden = options['n_hidden']
        i = T.nnet.sigmoid(_slice(preact, 0, n_hidden))
        f = T.nnet.sigmoid(_slice(preact, 1, n_hidden))
        o = T.nnet.sigmoid(_slice(preact, 2, n_hidden))
        c = T.tanh(_slice(preact, 3, n_hidden))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, theano_weights['lstm_W']) +
                   theano_weights['lstm_b'])

    n_hidden = options['n_hidden']
    rval, updates = theano.scan(
        _step,
        sequences=[mask, state_below],
        outputs_info=[
            T.alloc(np_floatX(0.), n_samples, n_hidden),
            T.alloc(np_floatX(0.), n_samples, n_hidden)],
        name='lstm_layers',
        n_steps=nsteps)

    return rval[0]


def sgd(lr, theano_weights, grads, x, mask, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in theano_weights.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(theano_weights.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, theano_weights, grads, x, mask, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    cost: Theano variable
        Objective function to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in theano_weights.items()]
    running_up2 = [theano.shared(p.get_value() * np_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in theano_weights.items()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in theano_weights.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(theano_weights.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, theano_weights, grads, x, mask, cost):
    """
    A variant of SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in theano_weights.items()]
    running_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in theano_weights.items()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in theano_weights.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * np_floatX(0.),
                           name='%s_updir' % k)
             for k, p in theano_weights.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(theano_weights.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def prepare_data(seqs):
    """ Create the matrices from the datasets.

    Parameters
    ----------
    seqs: list of sequences (or a single sequence)

    """
    if isinstance(seqs, np.ndarray):
        # Given a single sequence.
        assert seqs.ndim == 2
        seqs = [seqs]

    # Add 1 to length to account for the initial input vector.
    lengths = [len(s)+1 for s in seqs]

    dim = len(seqs[0][0])
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, dim)).astype('float64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[1:lengths[idx], idx, :] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask


cache = {}


def load_from_cache(model_options, new_theano_weights, new_state=None):
    cache_key = get_cache_key(**model_options)
    result = cache.get(cache_key, None)
    if result is None:
        return None

    new_state = {} if new_state is None else new_state

    functions, theano_weights, state = result

    new_functions = []
    for f in functions:
        exist_svs = set([i.variable for i in f.maker.inputs])
        swap_dict = {}
        for k, v in six.iteritems(theano_weights):
            if v in exist_svs:
                swap_dict[v] = new_theano_weights[k]
        for k, v in six.iteritems(state):
            if v in exist_svs:
                swap_dict[v] = new_state[k]
        new_functions.append(f.copy(swap=swap_dict))

    return new_functions


def truncate_roll(x, n=1):
    x = T.roll(x, -n, axis=0)
    x = T.set_subtensor(x[-n:], 0.0)
    return x


def store_in_cache(model_options, functions, theano_weights, state=None):
    if state is None:
        state = {}
    cache[get_cache_key(**model_options)] = (functions, theano_weights, state)


def make_verbose_print(verbosity, threshold=1.0):
    def vprint(*obj):
        if float(verbosity) >= float(threshold):
            print(*obj)
    return vprint


class TerminationDistribution(object):
    """ A continuous distribution augmented with a probability of terminating. """
    def __init__(self, dist, term_prob, term_symbol=None, random_state=None):
        self.dist = dist
        self.term_prob = term_prob
        self.term_symbol = term_symbol
        self.random_state = random_state

    def pdf(self, o):
        if o == self.term_symbol:
            return self.term_prob
        else:
            return (1 - self.term_prob) * self.dist.pdf(o)

    def rvs(self, size=None, random_state=None):
        random_state = (
            self.random_state if random_state is None else random_state)
        random_state = check_random_state(random_state)

        terminate = bernoulli(self.term_prob).rvs(size=size, random_state=random_state)

        try:
            int(terminate)
            if terminate:
                return self.term_symbol
            else:
                return self.dist.rvs(random_state=random_state)
        except:
            rvs = []
            for t in np.nditer(terminate):
                if t:
                    rvs.append(self.term_symbol)
                else:
                    v = self.dist.rvs(random_state=random_state)
                    rvs.append(v)
            rvs = np.array(rvs, dtype='object')
            rvs = rvs.reshape(*size)
            return rvs


class ProbabilisticRNN(SequenceModel):
    """ A ProbabilisticRNN. It can be used to assign probabilities
        to sequences of real-valued vectors.

    Parameters
    ----------
    n_input:
        Number of input units. For this type of RNN, the
        output has the same dimensionality as the input.
    n_hidden:
        Number of hidden units.
    patience:
        Number of epochs to wait before early stop if no progress occurs.
    max_epochs:
        The maximum number of epoch to run.
    dispFreq:
        Display to stdout the training progress every N updates
    lrate:
        Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer:
        sgd, adadelta and rmsprop available, sgd very hard to use, not
        recommended (probably need momentum and decaying learning rate).
    validFreq:
        Compute the validation error after this number of updates.
    saveFreq:
        Save the parameters after every saveFreq updates
    minibatch_size:
        The batch size during training.
    valid_minibatch_size:
        The batch size used for validation/test set.
    use_dropout: bool
        If False slightly faster, but worst test error.

    """

    term_symbol = None

    def __init__(
            self,
            n_input=2,
            n_hidden=5,
            recurrent_nl=T.nnet.sigmoid,
            patience=10,
            max_epochs=5000,
            dispFreq=1000,
            lrate=0.0001,
            optimizer=adadelta,
            validFreq=370,
            saveFreq=1110,
            minibatch_size=16,
            valid_minibatch_size=64,
            use_dropout=True,
            valid_pct=0.1,
            test_pct=0.0,
            use_cache=False,
            verbose=False,
            random_state=None):

        self.options = {}
        for k, v in six.iteritems(locals()):
            if k != 'self':
                setattr(self, k, v)
                self.options[k] = v

        theano.config.optimizer = "None"
        self.has_fit = False

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Reals(self.n_input, "ObsSpace")

    def check_terminal(self, o):
        if isinstance(o, np.ndarray):
            return False
        else:
            return o == self.term_symbol

    def _reset(self, initial=None):
        """ Set hidden state to initial hidden state. """
        self.theano_state['h_'].set_value(self.state['h_'])
        self.update(np.zeros(self.n_input))  # Fill in the hidden states
        self._cond_obs_dist = None

    def has_reward(self):
        return False

    def has_terminal_states(self):
        return True

    def update(self, o):
        """ Update the internal state upon seeing an observation. """
        self.update_hidden_state_(o)
        self._cond_obs_dist = None
        return self.theano_state['h_'].get_value()

    def cond_obs_prob(self, o):
        return self.cond_obs_dist().pdf(o)

    def cond_termination_prob(self):
        """ Get probability of terminating. """
        return self.cond_obs_dist().pdf(self.term_symbol)

    def cond_predict(self):
        return self.persist_pred_()

    def cond_predicts(self, string):
        """ Get predictions at each point in the string. """
        x, mask = prepare_data(string)
        predictions = self.f_pred_(x, mask)
        return predictions[:-1, 0, :]

    def cond_obs_dist(self):
        if self._cond_obs_dist is None:
            mean = self.persist_pred_()
            term_prob = float(self.persist_halt_())
            self._cond_obs_dist = TerminationDistribution(
                multivariate_normal(mean=mean), term_prob=term_prob)

        return self._cond_obs_dist

    def string_prob(self, string, log=False):
        x, mask = prepare_data(string)
        conditional_log_prob = self.f_log_prob_(x, mask)
        log_prob = conditional_log_prob.sum()
        return log_prob if log else np.exp(log_prob)

    def prefix_prob(self, string, log=False):
        x, mask = prepare_data(string)
        conditional_log_prob = self.f_log_prob_prefix_(x, mask)
        log_prob = conditional_log_prob.sum()
        return log_prob if log else np.exp(log_prob)

    def mean_log_likelihood(self, test_data, string=True):
        """ Get average log likelihood for the test data. """
        if not string:
            raise Exception("``string`` must be True.")
        if not test_data:
            return -np.inf

        x, mask = prepare_data(test_data)
        if string:
            conditional_log_prob = self.f_log_prob_(x, mask)
        else:
            conditional_log_prob = self.f_log_prob_prefix_(x, mask)
        return conditional_log_prob.sum() / len(test_data)

    def RMSE(self, test_data):
        n_predictions = sum(len(seq) for seq in test_data)
        if n_predictions == 0:
            return np.inf

        x, mask = prepare_data(test_data)
        se = self.f_squared_error_(x, mask)
        return np.sqrt(se.sum() / n_predictions)

    def deepcopy(self):
        raise NotImplementedError()

    @staticmethod
    def init_weights(options):
        weights = OrderedDict()

        n_hidden, n_input = options['n_hidden'], options['n_input']
        W = init_W(n_input, n_hidden)
        weights['rnn_W'] = W

        U = np.eye(n_hidden)  # Initializing to identity is supposed to work well with RELUs.
        # U = ortho_weight(n_hidden)
        weights['rnn_U'] = U

        b = np.zeros(n_hidden)
        weights['rnn_b'] = b.astype(config.floatX)

        weights['U'] = 0.01 * np.random.randn(
            options['n_hidden'], options['n_input']).astype(config.floatX)
        weights['b'] = np.zeros((options['n_input'],)).astype(config.floatX)

        weights['halt_U'] = 0.01 * np.random.randn(
            options['n_hidden'], 1).astype(config.floatX)
        weights['halt_b'] = np.zeros(1).astype(config.floatX)

        return weights

    @staticmethod
    def init_state(options):
        init_state = OrderedDict()

        init_state['use_noise'] = np_floatX(0.)
        init_state['h_'] = np_floatX(np.zeros(options['n_hidden']))

        return init_state

    @staticmethod
    def _build_model(theano_weights, theano_state, options):
        trng = RandomStreams(
            gen_seed(check_random_state(options['random_state'])))

        use_noise = theano_state['use_noise']  # Used for turning on/off dropout at train/test time.
        h_ = theano_state['h_']

        x = T.tensor3('x', dtype=config.floatX)
        mask = T.matrix('mask', dtype=config.floatX)

        proj = rnn_layer(theano_weights, x, options, mask=mask)

        if options['use_dropout']:
            proj = dropout_layer(proj, use_noise, trng)

        pred = T.dot(proj, theano_weights['U']) + theano_weights['b']
        halt = T.nnet.sigmoid(T.dot(proj, theano_weights['halt_U']) + theano_weights['halt_b'])

        rolled_mask = truncate_roll(mask, 1)
        sequence_end_mask = mask - rolled_mask
        rolled_x = truncate_roll(x, 1)

        squared_error = ((pred - rolled_x)**2) * rolled_mask[:, :, None]

        n_input = float(options['n_input'])

        # TODO: Implicitly assumes sigma == identity.
        log_prob_prefix = -0.5 * squared_error.sum(axis=2) - (n_input / 2) * np.log(2 * np.pi) * rolled_mask
        log_prob_prefix += (T.log(1 - halt) * (mask - sequence_end_mask)[:, :, None]).sum(axis=2)
        log_prob = log_prob_prefix + (T.log(halt) * sequence_end_mask[:, :, None]).sum(axis=2)

        f_log_prob_prefix = theano.function([x, mask], log_prob_prefix, name='f_log_prob_prefix')
        f_log_prob = theano.function([x, mask], log_prob, name='f_log_prob')
        f_squared_error = theano.function([x, mask], squared_error, name='f_squared_error')
        f_pred = theano.function([x, mask], pred, name='f_pred')
        f_hidden = theano.function([x, mask], proj, name='f_hidden')
        cost = -log_prob.sum()

        n_hidden = options['n_hidden']
        inp = T.vector()

        preact = T.dot(inp, theano_weights['rnn_W']) + theano_weights['rnn_b']
        preact += T.dot(h_, theano_weights['rnn_U'])

        h = options['recurrent_nl'](preact[:n_hidden])

        update_hidden_state = theano.function([inp], updates=[(h_, h)])

        if options['use_dropout']:
            proj = dropout_layer(h_, use_noise, trng)
        else:
            proj = h_

        persist_pred = theano.function([], T.dot(proj, theano_weights['U']) + theano_weights['b'])
        persist_halt = theano.function([], T.nnet.sigmoid(T.dot(proj, theano_weights['halt_U']) + theano_weights['halt_b']))

        return (x, mask, f_log_prob_prefix, f_log_prob, f_squared_error,
                f_pred, f_hidden, cost, halt, update_hidden_state,
                persist_pred, persist_halt)

    def _build_or_load(self):
        vprint = make_verbose_print(self.verbose)

        self.weights = weights = self.init_weights(self.options)
        self.theano_weights = theano_weights = as_shared_theano(weights)

        self.state = state = self.init_state(self.options)
        self.theano_state = theano_state = as_shared_theano(state)

        # Use only structural parameters in building the key.
        key_dict = dict(klass=self.__class__,
                        n_input=self.n_input,
                        n_hidden=self.n_hidden,
                        optimizer=self.optimizer,
                        use_dropout=self.use_dropout)

        vprint('Loading model from cache...')
        then = time.time()
        if self.use_cache:
            from_cache = load_from_cache(key_dict, theano_weights, theano_state)
        else:
            from_cache = None

        if from_cache is not None:
            vprint("Cache hit took %f seconds." % (time.time() - then))
            (f_log_prob_prefix, f_log_prob, f_squared_error,
             f_pred, f_hidden, f_cost, f_halt,
             f_grad, f_grad_shared, f_update,
             update_hidden_state, persist_pred, persist_halt, theano_state) = from_cache
        else:
            vprint("Cache miss took %f seconds." % (time.time() - then))

            vprint("Couldn't load the model from cache, building it from scratch.")
            then = time.time()
            (x, mask, f_log_prob_prefix, f_log_prob, f_squared_error,
             f_pred, f_hidden, cost, halt, update_hidden_state,
             persist_pred, persist_halt) = self._build_model(theano_weights, theano_state, self.options)

            f_cost = theano.function([x, mask], cost, name='f_cost')
            f_halt = theano.function([x, mask], halt, name='f_halt')

            grads = T.grad(cost, wrt=list(theano_weights.values()))
            f_grad = theano.function([x, mask], grads, name='f_grad')

            lr = T.scalar(name='lr')
            f_grad_shared, f_update = self.optimizer(lr, theano_weights, grads, x, mask, cost)
            vprint("Building model took %f seconds." % (time.time() - then))

            vprint("Storing model in cache...")
            then = time.time()
            functions = [
                f_log_prob_prefix, f_log_prob, f_squared_error,
                f_pred, f_hidden, f_cost, f_halt,
                f_grad, f_grad_shared, f_update,
                update_hidden_state, persist_pred, persist_halt]
            store_in_cache(key_dict, functions, theano_weights, theano_state)
            vprint("Storing model took %f seconds." % (time.time() - then))

        self.weights = weights
        self.theano_weights = theano_weights

        self.f_log_prob_prefix_ = f_log_prob_prefix
        self.f_log_prob_ = f_log_prob
        self.f_squared_error_ = f_squared_error
        self.f_pred_ = f_pred
        self.f_hidden_ = f_hidden
        self.f_cost_ = f_cost
        self.f_halt_ = f_halt

        self.f_grad_ = f_grad
        self.f_grad_shared_ = f_grad_shared
        self.f_update_ = f_update

        self.update_hidden_state_ = update_hidden_state
        self.persist_pred_ = persist_pred
        self.persist_halt_ = persist_halt

    def fit(self, train, valid=None, test=None, reuse=False, max_epochs=None):
        vprint = make_verbose_print(self.verbose)

        print("*" * 40)
        print("Beginning new fit for %s with reuse=%r." % (self.__class__.__name__, reuse))
        vprint("Options: ")
        vprint(str(self.options))

        valid_pct = self.valid_pct if valid is None else 0.0
        test_pct = self.test_pct if test is None else 0.0

        other_pct = valid_pct + test_pct

        if other_pct > 0.0:
            train, other = train_test_split(
                train, test_size=other_pct, random_state=self.random_state)

            if valid_pct > 0.0:
                n_valid = int(len(other) * valid_pct/(valid_pct + test_pct))
                valid = other[:n_valid]

            if test_pct > 0.0:
                n_test = int(len(other) * test_pct/(valid_pct + test_pct))
                test = other[-n_test:]

        assert valid, "Cannot proceed without validation data."
        test = [] if not bool(test) else test

        if not self.has_fit or not reuse:
            self._build_or_load()

        theano_weights = self.theano_weights

        f_grad_shared = self.f_grad_shared_
        f_update = self.f_update_

        vprint('Optimization')

        vprint("%d train examples" % len(train))
        vprint("%d valid examples" % len(valid))
        vprint("%d test examples" % len(test))

        history_errs = []
        best_weights = None
        bad_counter = 0

        validFreq = (len(train) // self.minibatch_size) if self.validFreq == -1 else self.validFreq
        # saveFreq = (len(train) // self.minibatch_size) if self.saveFreq == -1 else self.saveFreq

        first = True
        minibatch_idx = 0  # the number of updates done
        early_stop = False  # early stop
        start_time = time.time()
        try:
            for epoch_idx in range(self.max_epochs if max_epochs is None else max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(
                    len(train), self.minibatch_size, shuffle=True)

                for _, train_index in kf:
                    minibatch_idx += 1
                    self.theano_state['use_noise'].set_value(1.)

                    # Select the random examples for this minibatch
                    x = [train[t] for t in train_index]

                    # Return array with shape (maxlen_for_minibatch, n_samples)
                    x, mask = prepare_data(x)
                    n_samples += x.shape[1]

                    cost = f_grad_shared(x, mask)
                    f_update(self.lrate)

                    if np.isnan(cost) or np.isinf(cost):
                        raise Exception("Bad cost detected: %s." % cost)

                    if np.mod(minibatch_idx, self.dispFreq) == 0:
                        vprint('Epoch ', epoch_idx, 'Update ', minibatch_idx, 'Cost ', cost)

                    # if saveto and np.mod(minibatch_idx, saveFreq) == 0:
                    #     vprint('Saving...')

                    #     if best_weights is not None:
                    #         weights = best_weights
                    #     else:
                    #         weights = unzip(theano_weights)

                    #     np.savez(
                    #         saveto, history_errs=history_errs, **weights)
                    #     pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
                    #     vprint('Done')

                    if np.mod(minibatch_idx, validFreq) == 0:
                        self.theano_state['use_noise'].set_value(0.)
                        train_err = -self.mean_log_likelihood(train)
                        valid_err = -self.mean_log_likelihood(valid)
                        test_err = -self.mean_log_likelihood(test)

                        history_errs.append([valid_err, test_err])

                        if (best_weights is None or valid_err <= np.array(history_errs)[:, 0].min()):
                            best_weights = zip_weights(theano_weights)
                            bad_counter = 0

                        if first:
                            print("After first minibatch: ")
                            print('Train ', train_err,
                                  'Valid ', valid_err,
                                  'Test ', test_err)
                            first = False
                        else:
                            vprint(('Train ', train_err,
                                    'Valid ', valid_err,
                                    'Test ', test_err))

                        bad = len(history_errs) > self.patience
                        bad = bad and valid_err >= min(e[0] for e in history_errs[:-self.patience])
                        if bad:
                            bad_counter += 1
                            if bad_counter > self.patience:
                                vprint('Early Stop!')
                                early_stop = True
                                break

                if early_stop:
                    vprint('Triggered early stop!')
                    break

        except KeyboardInterrupt:
            vprint("Training interupted")

        end_time = time.time()
        if best_weights is not None:
            unzip_weights(best_weights, theano_weights)
        else:
            best_weights = zip_weights(theano_weights)

        self.theano_state['use_noise'].set_value(0.)
        train_err = -self.mean_log_likelihood(train)
        valid_err = -self.mean_log_likelihood(valid)
        test_err = -self.mean_log_likelihood(test)

        print("Final: ")
        print('Train ', train_err,
              'Valid ', valid_err,
              'Test ', test_err)

        # if saveto:
        #     np.savez(saveto, train_err=train_err,
        #              valid_err=valid_err, test_err=test_err,
        #              history_errs=history_errs, **best_weights)

        print('The code ran for %d epochs, with %f sec/epochs' % (
              (epoch_idx + 1), (end_time - start_time) / (1. * (epoch_idx + 1))))
        print(('Training took %.1fs' % (end_time - start_time)))
        print("*" * 40)

        self.has_fit = True

        self.reset()
        return self


class ProbabilisticGRU(ProbabilisticRNN):
    """ A ProbabilisticGRU. It can be used to assign probabilities
        to sequences of real-valued vectors.

    Parameters
    ----------
    See ProbabilisticRNN.

    """
    @staticmethod
    def init_weights(options):
        weights = OrderedDict()

        n_hidden, n_input = options['n_hidden'], options['n_input']
        W = np.concatenate([init_W(n_input, n_hidden),
                            init_W(n_input, n_hidden),
                            init_W(n_input, n_hidden)], axis=1)
        weights['gru_W'] = W

        U = np.concatenate([ortho_weight(n_hidden),
                            ortho_weight(n_hidden)], axis=1)
        weights['gru_U'] = U
        weights['gru_U_blank'] = ortho_weight(n_hidden)

        b = np.zeros(3 * n_hidden)
        weights['gru_b'] = b.astype(config.floatX)

        weights['U'] = 0.01 * np.random.randn(
            options['n_hidden'], options['n_input']).astype(config.floatX)
        weights['b'] = np.zeros((options['n_input'],)).astype(config.floatX)

        weights['halt_U'] = 0.01 * np.random.randn(
            options['n_hidden'], 1).astype(config.floatX)
        weights['halt_b'] = np.zeros(1).astype(config.floatX)

        return weights

    @staticmethod
    def _build_model(theano_weights, theano_state, options):
        trng = RandomStreams(
            gen_seed(check_random_state(options['random_state'])))

        use_noise = theano_state['use_noise']  # Used for turning on/off dropout at train/test time.
        h_ = theano_state['h_']

        x = T.tensor3('x', dtype=config.floatX)
        mask = T.matrix('mask', dtype=config.floatX)

        proj = gru_layer(theano_weights, x, options, mask=mask)

        if options['use_dropout']:
            proj = dropout_layer(proj, use_noise, trng)

        pred = T.dot(proj, theano_weights['U']) + theano_weights['b']
        halt = T.nnet.sigmoid(T.dot(proj, theano_weights['halt_U']) + theano_weights['halt_b'])

        rolled_mask = truncate_roll(mask, 1)
        sequence_end_mask = mask - rolled_mask
        rolled_x = truncate_roll(x, 1)

        squared_error = ((pred - rolled_x)**2) * rolled_mask[:, :, None]

        n_input = float(options['n_input'])

        # TODO: Implicitly assumes sigma == identity.
        log_prob_prefix = -0.5 * squared_error.sum(axis=2) - (n_input / 2) * np.log(2 * np.pi) * rolled_mask
        log_prob_prefix += (T.log(1 - halt) * (mask - sequence_end_mask)[:, :, None]).sum(axis=2)
        log_prob = log_prob_prefix + (T.log(halt) * sequence_end_mask[:, :, None]).sum(axis=2)

        f_log_prob_prefix = theano.function([x, mask], log_prob_prefix, name='f_log_prob_prefix')
        f_log_prob = theano.function([x, mask], log_prob, name='f_log_prob')
        f_squared_error = theano.function([x, mask], squared_error, name='f_squared_error')
        f_pred = theano.function([x, mask], pred, name='f_pred')
        f_hidden = theano.function([x, mask], proj, name='f_hidden')
        cost = -log_prob.sum()

        n_hidden = options['n_hidden']
        inp = T.vector()

        from_in = T.dot(inp, theano_weights['gru_W']) + theano_weights['gru_b']
        preact = T.dot(h_, theano_weights['gru_U']) + from_in[:2*n_hidden]

        r = T.nnet.sigmoid(preact[:n_hidden])
        u = T.nnet.sigmoid(preact[n_hidden:])

        h = T.tanh(T.dot(r*h_, theano_weights['gru_U_blank']) + from_in[2*n_hidden:])
        h = (1. - u) * h_ + u * h

        update_hidden_state = theano.function([inp], updates=[(h_, h)])

        if options['use_dropout']:
            proj = dropout_layer(h_, use_noise, trng)
        else:
            proj = h_

        persist_pred = theano.function([], T.dot(proj, theano_weights['U']) + theano_weights['b'])
        persist_halt = theano.function([], T.nnet.sigmoid(T.dot(proj, theano_weights['halt_U']) + theano_weights['halt_b']))

        return (x, mask, f_log_prob_prefix, f_log_prob, f_squared_error,
                f_pred, f_hidden, cost, halt, update_hidden_state,
                persist_pred, persist_halt)


class ProbabilisticLSTM(ProbabilisticRNN):
    """ A ProbabilisticLSTM. It can be used to assign probabilities
        to sequences of real-valued vectors.

    Parameters
    ----------
    See ProbabilisticRNN.

    """
    def _reset(self, initial=None):
        """ Set hidden state to initial hidden state. """
        self.theano_state['h_'].set_value(self.state['h_'])
        self.theano_state['c_'].set_value(self.state['c_'])
        self.update(np.zeros(self.n_input))  # Fill in the hidden states
        self._cond_obs_dist = None

    @staticmethod
    def init_weights(options):
        weights = OrderedDict()

        n_hidden, n_input = options['n_hidden'], options['n_input']
        W = np.concatenate([init_W(n_input, n_hidden),
                            init_W(n_input, n_hidden),
                            init_W(n_input, n_hidden),
                            init_W(n_input, n_hidden)], axis=1)
        weights['lstm_W'] = W

        U = np.concatenate([ortho_weight(n_hidden),
                            ortho_weight(n_hidden),
                            ortho_weight(n_hidden),
                            ortho_weight(n_hidden)], axis=1)
        weights['lstm_U'] = U

        b = np.zeros(4 * n_hidden)
        weights['lstm_b'] = b.astype(config.floatX)

        weights['U'] = 0.01 * np.random.randn(
            options['n_hidden'], options['n_input']).astype(config.floatX)
        weights['b'] = np.zeros((options['n_input'],)).astype(config.floatX)

        weights['halt_U'] = 0.01 * np.random.randn(
            options['n_hidden'], 1).astype(config.floatX)
        weights['halt_b'] = np.zeros(1).astype(config.floatX)

        return weights

    @staticmethod
    def init_state(options):
        init_state = OrderedDict()

        init_state['use_noise'] = np_floatX(0.)
        init_state['h_'] = np_floatX(np.zeros(options['n_hidden']))
        init_state['c_'] = np_floatX(np.zeros(options['n_hidden']))

        return init_state

    @staticmethod
    def _build_model(theano_weights, theano_state, options):
        trng = RandomStreams(
            gen_seed(check_random_state(options['random_state'])))

        use_noise = theano_state['use_noise']  # Used for turning on/off dropout at train/test time.
        h_ = theano_state['h_']
        c_ = theano_state['c_']

        x = T.tensor3('x', dtype=config.floatX)
        mask = T.matrix('mask', dtype=config.floatX)

        proj = lstm_layer(theano_weights, x, options, mask=mask)

        if options['use_dropout']:
            proj = dropout_layer(proj, use_noise, trng)

        pred = T.dot(proj, theano_weights['U']) + theano_weights['b']
        halt = T.nnet.sigmoid(T.dot(proj, theano_weights['halt_U']) + theano_weights['halt_b'])

        rolled_mask = truncate_roll(mask, 1)
        sequence_end_mask = mask - rolled_mask
        rolled_x = truncate_roll(x, 1)

        squared_error = ((pred - rolled_x)**2) * rolled_mask[:, :, None]

        n_input = float(options['n_input'])

        # TODO: Implicitly assumes sigma == identity.
        log_prob_prefix = -0.5 * squared_error.sum(axis=2) - (n_input / 2) * np.log(2 * np.pi) * rolled_mask
        log_prob_prefix += (T.log(1 - halt) * (mask - sequence_end_mask)[:, :, None]).sum(axis=2)
        log_prob = log_prob_prefix + (T.log(halt) * sequence_end_mask[:, :, None]).sum(axis=2)

        f_log_prob_prefix = theano.function([x, mask], log_prob_prefix, name='f_log_prob_prefix')
        f_log_prob = theano.function([x, mask], log_prob, name='f_log_prob')
        f_squared_error = theano.function([x, mask], squared_error, name='f_squared_error')
        f_pred = theano.function([x, mask], pred, name='f_pred')
        f_hidden = theano.function([x, mask], proj, name='f_hidden')
        cost = -log_prob.sum()

        n_hidden = options['n_hidden']
        inp = T.vector()

        preact = T.dot(inp, theano_weights['lstm_W']) + theano_weights['lstm_b']
        preact += T.dot(h_, theano_weights['lstm_U'])

        i = T.nnet.sigmoid(preact[:n_hidden])
        f = T.nnet.sigmoid(preact[n_hidden:2*n_hidden])
        o = T.nnet.sigmoid(preact[2*n_hidden:3*n_hidden])
        c = T.tanh(preact[3*n_hidden:])
        c = f * c_ + i * c
        h = o * T.tanh(c)

        update_hidden_state = theano.function([inp], updates=[(h_, h), (c_, c)])

        if options['use_dropout']:
            proj = dropout_layer(h_, use_noise, trng)
        else:
            proj = h_

        persist_pred = theano.function([], T.dot(proj, theano_weights['U']) + theano_weights['b'])
        persist_halt = theano.function([], T.nnet.sigmoid(T.dot(proj, theano_weights['halt_U']) + theano_weights['halt_b']))

        return (x, mask, f_log_prob_prefix, f_log_prob, f_squared_error,
                f_pred, f_hidden, cost, halt, update_hidden_state,
                persist_pred, persist_halt)


def sequences_from_predictions(preds, mask):
    seqs = [[] for i in range(preds.shape[1])]
    for j in range(preds.shape[0]):
        for i in range(preds.shape[1]):
            if mask[j, i] > 0:
                seqs[i].append(preds[j, i, :])
    return seqs


def demo(lstm, data, labels):
    # Find an example of each digit, plot how the network does on each.
    unique_labels = list(set(labels))

    digits_to_plot = []
    for l in unique_labels:
        for i in np.random.permutation(range(len(labels))):
            if labels[i] == l:
                digits_to_plot.append(data[i])
                break

    x, mask = prepare_data(digits_to_plot)
    preds = lstm.f_pred_(x, mask)

    seqs = sequences_from_predictions(preds, mask)

    import matplotlib.pyplot as plt
    for i, digit in enumerate(digits_to_plot):
        plt.subplot(2, 1, 1)
        plot_digit(digit, difference)
        plt.title("Ground Truth: Label = %d" % unique_labels[i])
        plt.subplot(2, 1, 2)
        plt.title("Prediction")
        plot_digit(seqs[i], difference)
        plt.show()


if __name__ == "__main__":
    from spectral_dagger.datasets.pendigits import get_data, plot_digit

    difference = True
    sample_every = 1
    _data, _labels = get_data(difference=difference, sample_every=sample_every, ignore_multisegment=False)

    data = []
    labels = []
    for ds, ls in zip(_data, _labels):
        for d, l in zip(ds, ls):
            data.append(d)
            labels.append(l)
    permutation = np.random.permutation(range(len(labels)))
    data = [data[i] for i in permutation]
    labels = [labels[i] for i in permutation]

    n_training_samples = 30
    n_test_samples = 30
    test_data = data[n_training_samples:n_training_samples+n_test_samples]
    training_data = data[:n_training_samples]
    test_labels = labels[n_training_samples:n_training_samples+n_test_samples]
    training_labels = labels[:n_training_samples]

    max_length = None
    model_class = ProbabilisticGRU if 1 else (ProbabilisticRNN if 1 else ProbabilisticLSTM)
    verbose = True
    use_dropout = True
    quick = 0
    if quick:
        model = model_class(n_hidden=2, max_epochs=1, use_dropout=use_dropout, verbose=verbose)
    else:
        model = model_class(n_hidden=5, max_epochs=10, use_dropout=use_dropout, verbose=verbose)

    model.fit(training_data, max_epochs=2)

    # demo(model, training_data, training_labels)
    # demo(model, test_data, test_labels)

    model.fit(training_data)

    print("Mean Log likelihood of training data: %g" % model.mean_log_likelihood(training_data))
    print("Mean Log likelihood of test data: %g" % model.mean_log_likelihood(test_data))

    for i in range(2):
        for sequence in training_data[:1]:
            if max_length is not None:
                sequence = sequence[:max_length]

            print("Reset:")
            log_prob = 0.0
            model.reset()
            for i, s in enumerate(sequence):
                print("S: ", s)
                log_prob += np.log(model.cond_obs_prob(s))
                print("Mean: ", model.cond_obs_dist().dist.mean)
                print("Log prob: ", log_prob)
                new_hidden = model.update(s)
                print("New hidden: ", new_hidden)

            x, mask = prepare_data(sequence)
            y = model.f_hidden_(x, mask)
            print("HIDDEN GROUND TRUTH: ", y)

            print("Prefix: ", log_prob)
            print("Prefix: ", model.prefix_prob(sequence, log=True))
            log_prob += np.log(model.cond_termination_prob())
            print("String: ", log_prob)
            print("String: ", model.mean_log_likelihood([sequence]))
            print("String: ", model.string_prob(sequence, log=True))
        model.fit(training_data, max_epochs=1000, reuse=True)
