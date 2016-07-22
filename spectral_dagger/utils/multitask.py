import six


class MultitaskSequenceDataset(object):
    """ An object containing data for multiple tasks.

    Can be used as the ``X`` in the Dataset classes (and the ``y`` as well
    for the case of supervised learning).

    Either set of indices can be omitted. In that case, the core datasets
    are automatically given indices range(len(core)), and transfer datasets
    are given indices beginning at one more than the largest core index.

    The difference between core and transfer is only really relevant at
    training time.

    Parameters
    ----------
    core: array-like
        Core data.
    core_indices: list of int
        Indices/labels for the datasets in ``core``.
    transfer: array-like
        transfer data.
    transfer_indices: list of int
        Indices/labels for the datasets in ``transfer``.
    context: dict
        Added as attributes.

    """
    def __init__(
            self, core=None, core_indices=None,
            transfer=None, transfer_indices=None, **context):

        for k, v in six.iteritems(context):
            setattr(self, k, v)

        self.core_data = [] if core is None else core

        if core_indices is None:
            core_indices = range(len(self.core_data))
        self.core_indices = core_indices
        assert len(self.core_data) == len(self.core_indices)

        self.transfer_data = [] if transfer is None else transfer
        if transfer_indices is None:
            start = max(self.core_indices) if self.core_indices else 0
            transfer_indices = range(start, start + len(self.transfer_data))
        self.transfer_indices = transfer_indices
        assert len(self.transfer_data) == len(self.transfer_indices)

        self.context = context

        self.shape = (max(len(cd) for cd in self.core_data),)

    @property
    def X(self):
        """ Having this property makes this class act like a ``Dataset``. """
        return self

    @property
    def y(self):
        """ Having this property makes this class act like a ``Dataset``. """
        return None

    def __len__(self):
        return len(self.core_data[0])

    def __getitem__(self, key):
        try:
            key = list(key)
            core_data = [[c[k] for k in key] for c in self.core_data]
            transfer_data = [[t[k] for k in key] for t in self.transfer_data]
        except TypeError:
            core_data = [c[key] for c in self.core_data]
            transfer_data = [t[key] for t in self.transfer_data]

        return MultitaskSequenceDataset(
            core=core_data, transfer=transfer_data,
            core_indices=self.core_indices,
            transfer_indices=self.transfer_indices,
            **self.context)

    @property
    def data(self):
        return self.core_data + self.transfer_data

    @property
    def indices(self):
        return self.core_indices + self.transfer_indices

    @property
    def n_core(self):
        return len(self.core)

    @property
    def n_transfer(self):
        return len(self.transfer)

    @property
    def core(self):
        return zip(self.core_indices, self.core_data)

    @property
    def transfer(self):
        return zip(self.transfer_indices, self.transfer_data)
