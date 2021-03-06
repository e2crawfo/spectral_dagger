import pandas as pd
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap
import numpy as np
import warnings
import logging
from itertools import product
import collections

logger = logging.getLogger(__name__)


def plot_measures(
        results, measures, x_var, split_var,
        measure_display=None, x_var_display=None, legend_loc='right',
        errorbars=True, logx=None, logy=None, jitter_x=None, kwarg_func=None,
        kwargs=None, fig=None):
    """ A function for plotting certain pandas DataFrames.

    Assumes the ``results`` data frame has a certain format. In particular,
    each row in ``results`` should correspond to a single trial or simulation.
    Each field/column should do one of two things: describe the context of the
    trial (such as one of the independent variables the trial was run with),
    or record some aspect of the result of the trial.

    ``results`` is expected to have simple integer indices, each
    trial/simulation having its own index.

    ``measures`` should be a list of names of result fields in ``results``.
    For each entry in ``measures``, this function will create a separate plot,
    with the plots stacked vertically in the figure.

    ``x_var`` should be a string giving the name of one of the context fields
    in ``results`` which will be used as the x-axis for all plots.

    ``split_var`` should be a string or list of strings, each string giving
    the name of one of the context fields in ``results``. For each unique
    value of ``split_var`` found in ``results`` (or combination of values,
    in the case that a list of strings is supplied), a separate line will be
    created in each of the plots. All trials which have identical values for
    both ``x_var`` and ``split_var`` will be averaged over, and the mean will
    be plotted along with bootstrapped 95% confidence intervals.

    Parameters
    ----------
    results: str or DataFrame
        Data to plot. Each row is a trial. If a string, assumed to be the name
        of a csv file which will be loaded from disk.
    measures: str or list of str
        Create a separate subplot for each entry.
    x_var: str
        Name of the variable to plot along x-axis.
    split_var: str or (list of str)
        Each in the plots corresponds to a unique value of the ``split_var``
        field in results.
    measure_display: list of str (optional)
        A list of strings for displaying as the y-axis labels for the different
        measures. Should have the same length as ``measures``.
    x_var_display: str (optional)
        A string for displaying as the x-axis label.
    legend_loc: one of 'left', 'bottom' (optional)
        Where to place the legend.
    errorbars: (optional)
        'fill' -> Will calculate error bars and display them with transparent filling.
        Any other truthy value -> Will calculate error bars and display them normally.
        Any falsy value -> No error bars.
    logx: float > 1 or None (optional)
        If None, x-axis scaled as normal. Otherwise, scaled logarithmically
        using ``logx`` as base.
    logy: float > 1 or None (optional)
        Similar to logx.
    jitter_x: float > 0.0 or None (optional)
        Standard deviation of noise to add to x-values to avoid overlapping points.
        If < 0, then uses deterministic jittering with ``-jitter_x`` as width.
    kwarg_func: func (optional)
        A function which accepts one of the values from ``split_var`` and
        returns a dict of key word arguments for the call to plt.errorbar
        for that value of ``split_var``.
    kwargs: dict
        Additional key word args for used for every call to plt.errorbar.
        Arguments obtained by calling ``kwarg_func`` will overwrite the
        args in ``kwargs`` if there is a conflict.
    fig: matplotlib Figure (optional)
        A figure to plot on. If not provided, current figure is used.

    """
    if isinstance(results, str):
        results = pd.read_csv(results)
    if isinstance(measures, str):
        measures = [measures]
    if isinstance(measure_display, str):
        measure_display = [measure_display]
    if isinstance(split_var, str):
        split_var = [split_var]

    if fig is None:
        fig = plt.gcf()

    logger.info(results)
    logger.info(results.describe())

    fig.subplots_adjust(left=0.1, right=0.75, top=0.8, bottom=0.1)

    n_plots = len(measures)
    axes = []
    legend_handles = {}
    for i, measure in enumerate(measures):
        try:
            measure_str = measure_display[i]
        except Exception:
            measure_str = measure

        ax = fig.add_subplot(n_plots, 1, i+1)
        _plot_measure(
            results, measure, x_var, split_var, measure_display=measure_str,
            logx=logx, logy=logy, jitter_x=jitter_x, kwarg_func=kwarg_func,
            kwargs=kwargs, ax=ax, errorbars=errorbars)

        legend_handles.update(
            **{l: h for h, l in zip(*ax.get_legend_handles_labels())})

        axes.append(ax)

    ordered_labels = sorted(legend_handles.keys())
    ordered_handles = [legend_handles[l] for l in ordered_labels]

    if legend_loc == 'right_side':
        # ax.legend(
        #     loc='center left', bbox_to_anchor=(1, 0.5),
        #     prop={'size': 10}, handlelength=3.0, handletextpad=.5,
        #     shadow=False, frameon=False)
        fig.legend(
            ordered_handles, ordered_labels, loc='center left',
            bbox_to_anchor=(0.0, 0.5), ncol=1)
    elif legend_loc == 'bottom':
        fig.legend(
            ordered_handles, ordered_labels, loc='lower center',
            bbox_to_anchor=(0.5, 0.0), ncol=4)
    elif legend_loc is not None:
        fig.legend(ordered_handles, ordered_labels, loc=legend_loc)

    ax.set_xlabel(x_var if x_var_display is None else x_var_display)

    return fig, axes


def jitter(arr, s):
    _range = max(arr)-min(arr)
    _range = _range if _range else 1.0
    stdev = s * _range
    return arr + np.random.randn(len(arr)) * stdev


def deterministic_jitter(arr, i, n, s=0.25):
    if n == 1:
        return arr
    assert n > 1 and isinstance(n, int)
    assert s >= 0.0

    if len(arr) == 1:
        return arr + np.linspace(-s, s)[i]

    spacing = max(arr[1:]-arr[:-1])
    delta = (s * spacing * np.linspace(-0.5, 0.5, n))[i]
    return arr + delta


def _plot_measure(
        results, measure, x_var, split_vars, measure_display=None,
        errorbars=True, logx=None, logy=None, jitter_x=None, kwarg_func=None,
        kwargs=None, ax=None):

    if ax is None:
        ax = plt.gca()

    if kwargs is None:
        kwargs = {}

    measure_data = results[split_vars + [x_var, measure]]
    grouped = measure_data.groupby(split_vars + [x_var])
    mean = grouped.mean()
    logger.info(mean)

    if errorbars:
        ci_lower = pd.Series(data=0.0, index=mean.index)
        ci_upper = pd.Series(data=0.0, index=mean.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for name, group in grouped:
                values = group[measure].values
                values = values[np.logical_not(np.isnan(values)) & np.logical_not(np.isinf(values))]

                if len(values) > 1:
                    try:
                        ci = bootstrap.ci(values)
                    except Exception:
                        ci = values[0], values[0]

                elif len(values) == 1:
                    ci = values[0], values[0]
                else:
                    ci = (np.nan, np.nan)

                ci_lower[name] = ci[0]
                ci_upper[name] = ci[1]

        mean['ci_lower'] = ci_lower
        mean['ci_upper'] = ci_upper

    mean = mean.reset_index()
    mean = mean[
        np.logical_not(np.isnan(mean[measure].values)) &
        np.logical_not(np.isinf(mean[measure].values))]

    sv_series = pd.Series(
        list(zip(*[mean[sv] for sv in split_vars])), index=mean.index)

    if logx is not None:
        ax.set_xscale("log", nonposx='clip', basex=logx)
    if logy is not None:
        ax.set_yscale("log", nonposy='clip', basey=logy)

    lines = []  # List of lines that are added, doesn't include errorbars.

    X_values = []
    Y_values = []

    sv_unique = list(sv_series.unique())
    n = len(sv_unique)
    legend_handles_labels = []
    for i, sv in enumerate(sv_unique):
        data = mean[sv_series == sv]

        X = data[x_var].values
        if jitter_x is not None:
            if jitter_x < 0:
                X = deterministic_jitter(X, i, n, -jitter_x)
            else:
                X = jitter(X, jitter_x)

        Y = data[measure].values

        X_values.extend(X)
        Y_values.extend(Y)

        _kwargs = kwargs.copy()
        if isinstance(kwarg_func, collections.Callable):
            if len(sv) == 1:
                sv = sv[0]
            _kwargs.update(kwarg_func(sv))
        else:
            _kwargs.update(dict(label=sv))

        if errorbars == 'fill':
            y_lower = data['ci_lower'].values
            y_upper = data['ci_upper'].values
            Y_values.extend(data['ci_lower'].values)
            Y_values.extend(data['ci_upper'].values)

            label = _kwargs.pop('label', '')
            l = ax.plot(X, Y, **_kwargs)
            l = l[0]

            fill_kwargs = {'facecolor': l.get_color(), 'edgecolor': l.get_color(), 'alpha': 0.3, 'label': label}
            ax.fill_between(X, y_lower, y_upper, **fill_kwargs)
            fill = ax.fill(np.NaN, np.NaN, color=l.get_color(), alpha=0.5)
            legend_handles_labels.append(((l, fill[0]), label))

        elif errorbars:
            y_lower = data[measure].values - data['ci_lower'].values
            y_upper = data['ci_upper'].values - data[measure].values
            Y_values.extend(data['ci_lower'].values)
            Y_values.extend(data['ci_upper'].values)
            yerr = np.vstack((y_lower, y_upper))
            label = _kwargs.get('label', '')

            l = ax.errorbar(X, Y, yerr=yerr, **_kwargs)
            legend_handles_labels.append((l[0], label))

        else:
            label = _kwargs.get('label', '')
            l = ax.plot(X, Y, **_kwargs)
            legend_handles_labels.append((l[0], label))

        lines.append(l)

    ax._legend_handles_labels = zip(*legend_handles_labels)

    def _remove_inf(s):
        if isinstance(s, pd.Series):
            return s.replace([np.inf, -np.inf], np.nan)
        else:
            s = np.array(s, dtype='f')
            s[np.isinf(s)] = np.nan
            return s

    lo_x = _remove_inf(X_values).min()
    hi_x = _remove_inf(X_values).max()

    xlim_lo = lo_x - 0.05 * (hi_x - lo_x)
    xlim_hi = hi_x + 0.05 * (hi_x - lo_x)
    ax.set_xlim(xlim_lo, xlim_hi)

    lo_y = _remove_inf(Y_values).min()
    hi_y = _remove_inf(Y_values).max()

    ylim_lo = lo_y - 0.05 * (hi_y - lo_y)
    ylim_hi = hi_y + 0.05 * (hi_y - lo_y)
    ax.set_ylim(ylim_lo, ylim_hi)

    ax.set_ylabel(measure if measure_display is None else measure_display)

    return ax, lines


def single_split_var(display=False):
    # An example of using plot_measures
    epsilon = 0.5
    x_values = np.linspace(-2, 2, 10)
    results = []

    funcs = [lambda a: a**2, lambda a: a]
    func_names = ['Quadratic', 'Linear']
    n_repeats = 10
    rng = np.random.RandomState(10)

    iterator = product(
        list(range(n_repeats)), x_values, list(zip(funcs, func_names)))

    for i, x, (f, name) in iterator:
        y = f(x) + epsilon * rng.normal()
        y_squared = f(x)**2 + epsilon * rng.normal()
        results.append(dict(
            name=name, x=x, y=y, negative_y=-y,
            y_squared=y_squared))

    results = pd.DataFrame.from_records(results)

    def kwarg_func(split_var):
        label = "Function: %s" % split_var
        if split_var == 'Quadratic':
            return dict(label=label, linestyle='-')
        else:
            return dict(label=label, linestyle='--')

    plot_measures(
        results, measures=['y', 'negative_y', 'y_squared'],
        x_var='x', split_var='name', kwarg_func=kwarg_func)

    if display:
        plt.show()


def multiple_split_vars(display=False):
    # An example of using plot_measures
    epsilon = [0.5, 2.0]
    x_values = np.linspace(-2, 2, 10)
    results = []

    funcs = [lambda a: a**2, lambda a: a]
    func_names = ['Quadratic', 'Linear']
    n_repeats = 10
    rng = np.random.RandomState(10)

    iterator = product(
        list(range(n_repeats)), x_values, list(zip(funcs, func_names)), epsilon)

    for i, x, (f, name), e in iterator:
        y = f(x) + e * rng.normal()
        y_squared = f(x)**2 + e * rng.normal()
        results.append(dict(
            name=name, x=x, y=y, negative_y=-y,
            y_squared=y_squared, epsilon=e))

    results = pd.DataFrame.from_records(results)

    def kwarg_func(split_var):
        name, epsilon = split_var

        label = "Function: %s, Noise Level: %f" % (name, epsilon)
        if name == 'Quadratic':
            return dict(label=label, linestyle='-')
        else:
            return dict(label=label, linestyle='--')

    plot_measures(
        results, measures=['y', 'negative_y', 'y_squared'],
        x_var='x', split_var=['name', 'epsilon'], kwarg_func=kwarg_func)

    if display:
        plt.show()

if __name__ == "__main__":
    single_split_var(True)
    multiple_split_vars(True)
