""" Core functions used as kernels of inverse transformation sampling
"""
import numpy as np
from astropy.utils import NumpyRNGContext
from warnings import warn
import warnings

from .array_utils import unsorting_indices


__all__ = ('monte_carlo_from_cdf_lookup', 'build_cdf_lookup', 'rank_order_percentile')

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def monte_carlo_from_cdf_lookup(x_table, y_table, mc_input='random',
        num_draws=None, seed=None):
    """
    Randomly draw a set of ``num_draws`` points from any arbitrary input distribution function.
    The input distribution is specified in terms of its CDF,
    which is defined by the values of the input ``y_table`` of the CDF
    evaluated at an input set of control points ``x_table``.
    The input ``x_table`` and ``y_table`` can be calculated from any input dataset
    using the function `build_cdf_lookup`.

    Parameters
    ----------
    x_table : ndarray
        Numpy array of shape (npts, ) providing the control points
        at which the input CDF has been tabulated.

    y_table : ndarray
        Numpy array of shape (npts, ) providing the values of the random variable
        associated with each input control point.

    mc_input : ndarray, optional
        Input array of shape (num_desired_draws, ) storing values between 0 and 1
        specifying the values of the CDF for which associated values of ``y`` are desired.
        If ``mc_input`` is left unspecified, the ``num_draws`` must be specified;
        in this case, the CDF will be randomly sampled.

    num_draws : int, optional
        Desired number of random draws from the input CDF.
        ``num_draws`` must be specified if ``mc_input`` is left unspecified.

    seed : int, optional
        Random number seed used in the Monte Carlo realization.

    Returns
    --------
    mc_realization : ndarray
        Length-num_draws array of random draws from the input CDF.

    Notes
    -----
    See the `Transformation of Probability tutorial <https://github.com/jbailinua/probability/>`_
    for pedagogical derivations associated with inverse transformation sampling.

    Examples
    --------
    In this first example, we'll start by creating some fake data, ``y``,
    drawn from a weighted combination of a Gaussian and a power law.
    We'll think of the fake data ``y`` as if it came from some
    external dataset that we want to model, treating the fake data as our
    input distribution. We will use the `build_cdf_lookup` function
    to build the necessary lookup tables ``x_table`` and ``y_table``,
    and then use the `monte_carlo_from_cdf_lookup` function
    to generate a Monte Carlo realization of the data ``y``.

    >>> npts = int(1e4)
    >>> y = 0.1*np.random.normal(size=npts) + 0.9*np.random.power(2, size=npts)
    >>> x_table, y_table = build_cdf_lookup(y)
    >>> result = monte_carlo_from_cdf_lookup(x_table, y_table, num_draws=5000)

    The returned array ``result`` is a stochastic Monte Carlo realization of the
    distribution specified by ``y``.

    .. image:: /_static/monte_carlo_example.png

    Now let's consider a second example where, rather than specifying an integer number of
    purely random Monte Carlo draws, instead we pass in an array of uniform random numbers.

    >>> uniform_randoms = np.random.rand(npts)
    >>> result2 = monte_carlo_from_cdf_lookup(x_table, y_table, mc_input=uniform_randoms)

    This alternative call to `monte_carlo_from_cdf_lookup` provides completely
    equivalent results as the first, because we have passed in uniform randoms.
    Since ``uniform_randoms`` just stores random values, then random values
    from the input distribution will be returned.

    However, this alternative form of input can be exploited for other applications.
    Suppose that instead of purely random draws, you wish to draw from the input distribution ``y``
    in such a way that you introduce a correlation between the drawn values and the values stored
    in some other array, ``x``. You can accomplish this by passing in the rank-order
    percentile values of ``x`` instead of uniform randoms.
    This is the basis of the conditional abundance matching technique
    implemented by the `~halotools.empirical_models.conditional_abunmatch_bin_based` function.

    To see an example of how this works, let's create some fake data for some property *x*
    that we wish to model as being correlated with Monte Carlo realizations of *y* while
    preserving the PDF of the realization:

    >>> x = np.sort(10**np.random.normal(loc=10, size=5000))
    >>> x_percentile = (1. + np.arange(len(x)))/float(len(x)+1)
    >>> correlated_result = monte_carlo_from_cdf_lookup(x_table, y_table, mc_input=x_percentile)
    >>> uniform_randoms = np.random.rand(npts)
    >>> uncorrelated_result = monte_carlo_from_cdf_lookup(x_table, y_table, mc_input=uniform_randoms)

    We can use the `~halotools.empirical_models.noisy_percentile` to introduce variable levels
    of noise in the correlation.

    >>> from halotools.empirical_models import noisy_percentile
    >>> noisy_x_percentile = noisy_percentile(x_percentile, correlation_coeff=0.75)
    >>> weakly_correlated_result = monte_carlo_from_cdf_lookup(x_table, y_table, mc_input=noisy_x_percentile)

    .. image:: /_static/monte_carlo_example2.png

    """
    if mc_input is 'random':
        if num_draws is None:
            msg = ("If input ``mc_input`` is set to ``random``, \n"
                "``num_draws`` must be specified")
            raise ValueError(msg)
        with NumpyRNGContext(seed):
            mc_input = np.random.rand(num_draws)
    else:
        if num_draws is not None:
            msg = ("If input ``mc_input`` is specified, \n"
                "the ``num_draws`` keyword should be left unspecified.")
            raise ValueError(msg)

    return np.interp(np.atleast_1d(mc_input), x_table, y_table)


def build_cdf_lookup(y, npts_lookup_table=1000):
    r""" Compute a lookup table for the cumulative distribution function
    specified by the input set of ``y`` values.

    The input data ``y`` will be used to define the CDF P(< y) in the usual way:
    the array ``y`` will be sorted, and the largest value corresponds to CDF value 1/npts_data,
    the second largest value 2/npts_data, etc. For performance reasons,
    this correspondence will be used to build a sparse lookup table
    of length ``npts_lookup_table``. The accuracy of the returned
    CDF is fundamentally limited by npts_data, and optionally limited by npts_lookup_table.

    Parameters
    ----------
    y : ndarray
        Numpy array of shape (npts_data, ) defining the distribution function
        of the returned lookup table.

    npts_lookup_table : int, optional
        Number of control points in the returned lookup table. Cannot exceed npts_data.

    Returns
    -------
    x_table : ndarray
        Numpy array of shape (npts_lookup_table, ) storing the control points
        at which the CDF has been evaluated.

    y_table : ndarray
        Numpy array of shape (npts_lookup_table, ) storing the values of the
        random variable associated with each control point in the ``x_table``.

    Examples
    --------
    >>> y = np.random.normal(size=int(1e5))
    >>> x_table, y_table = build_cdf_lookup(y, npts_lookup_table=100)
    """
    y = np.atleast_1d(y)
    assert len(y) > 1, "Input ``y`` has only one element"

    npts_y = len(y)
    if npts_y < npts_lookup_table:
        warning_msg = ("The build_cdf_lookup function was called with the (optional) ``npts_lookup_table`` "
            "argument set to {0}.\n"
            "However, the number of data points in your data table npts_y = {1}.\n"
            "The default behavior in this situation is to overwrite ``npts_lookup_table`` = ``npts_y``,\n"
            "so that every point in the data set is used to build the lookup table.")
        warn(warning_msg.format(npts_lookup_table, npts_y))
    npts_lookup_table = max(npts_lookup_table, npts_y)

    sorted_y = np.sort(y)
    normed_rank_order = _sorted_rank_order_array(npts_y)
    x_table = _sorted_rank_order_array(npts_lookup_table)
    y_table = np.interp(x_table, normed_rank_order, sorted_y)
    return x_table, y_table


def rank_order_percentile(y):
    """ Return the rank-order percentile of the values of an input distribution ``y``.

    Parameters
    ----------
    y : ndarray
        Numpy array of shape (npts, ) storing an input distribution of values

    Returns
    -------
    ranks : ndarray
        Numpy array of shape (npts, ) storing the rank-order percentile
        of every value of the input ``y``.

    Examples
    --------
    >>> percentile = rank_order_percentile(np.random.normal(size=int(1e2)))
    """
    idx_sorted = np.argsort(y)
    return _sorted_rank_order_array(len(y))[unsorting_indices(idx_sorted)]


def _sorted_rank_order_array(npts):
    """ Return the length-npts array [1/npts+1, 2/npts+1, ... npts/npts+1].
    Values are linearly spaced in ascending order respecting the strict
    inequalities 0 < x < 1.

    Notes
    -----
    This function is typically used to calculate the control points of a CDF.
    """
    return np.arange(1, npts+1)/float(npts+1)
