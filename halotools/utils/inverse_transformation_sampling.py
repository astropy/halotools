""" Core functions used as kernels of inverse transformation sampling
"""
import numpy as np
from astropy.utils import NumpyRNGContext
from warnings import warn

from .array_utils import unsorting_indices


__all__ = ('monte_carlo_from_cdf_lookup', 'build_cdf_lookup', 'rank_order_percentile')


def monte_carlo_from_cdf_lookup(x_table, y_table, mc_input='random',
        num_draws=None, seed=None):
    """
    Randomly draw a set of ``num_draws`` points from an input distribution function.
    The input distribution is specified in terms of its CDF,
    which is defined by the values of the input ``y_table`` of the CDF
    evaluated at an input set of control points ``x_table``.

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
    The function `build_cdf_lookup` can be used to generate the two input tables
    ``x_table`` and ``y_table`` starting from an input distribution.

    Examples
    --------
    Below we'll generate some fake data drawn from a weighted combination of a
    Gaussian and a power law, build the necessary lookup tables and then
    use the `monte_carlo_from_cdf_lookup` function to
    randomly resample the input distribution.

    >>> npts = int(1e4)
    >>> y = 0.3*np.random.normal(size=npts) + 0.7*np.random.power(2, size=npts)
    >>> x_table, y_table = build_cdf_lookup(y)
    >>> result = monte_carlo_from_cdf_lookup(x_table, y_table, num_draws=100)

    Now suppose we want to draw from this tabulate distribution, associating
    each draw with a desired input rank-ordered percentile (as is the case,
    for example, in applications of conditional abundance matching).

    >>> result = monte_carlo_from_cdf_lookup(x_table, y_table, mc_input=np.array((0.1, 0.4, 0.25)))

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
    """ Compute a lookup table for the cumulative distribution function
    specified by the input set of ``y`` values.

    Parameters
    ----------
    y : ndarray
        Numpy array of shape (npts_data, ) defining the distribution function
        of the returned lookup table.

    npts_lookup_table : int, optional
        Number of control points in the returned lookup table.

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
        warn("npts_y = {0} is less than  npts_lookup_table = {1}.\n"
            "Setting npts_lookup_table to {0}".format(npts_y, npts_lookup_table))
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
