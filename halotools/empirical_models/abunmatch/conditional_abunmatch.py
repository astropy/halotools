""" Module storing the Numpy kernel for conditional abundance matching
"""
import numpy as np
from astropy.utils import NumpyRNGContext
from ...utils import inverse_transformation_sampling as its
from ...utils import unsorting_indices


__author__ = ('Andrew Hearin', 'Duncan Campbell')
__all__ = ('conditional_abunmatch', 'randomly_resort')


def conditional_abunmatch(x, y, sigma=0., npts_lookup_table=1000, seed=None):
    """ Function used to model a correlation between two variables ``x`` and ``y``
    using the conditional abundance matching technique.

    Parameters
    -----------
    x : ndarray
        Numpy array of shape (npts_x, ) typically storing a halo property

    y : ndarray
        Numpy array of shape (npts_y, ) typically storing a galaxy property

    sigma : float, optional
        Variable controlling the level of Gaussian noise that will be introduced
        to the x--y correlation. Default is 0, for a perfect monotonic relation
        between x and y.

    npts_lookup_table : int, optional
        Size of the lookup table used to approximate the ``y`` distribution.
        Default is 1000.

    seed : int, optional
        Random number seed used to introduce noise in the x--y correlation.
        Default is None for stochastic results.

    Returns
    -------
    modeled_y : ndarray
        Numpy array of shape (npts_x, ) storing the modeled y-values associated
        with each value of the input ``x``.

    Examples
    --------
    Suppose we would like to do some CAM-style modeling of a correlation between some
    halo property ``x`` and some galaxy property ``y``. The `conditional_abunmatch` function
    can be used to map values of the galaxy property onto the halos in such a way that the
    PDF of ``y`` is preserved and a correlation (of variable strength)
    between ``x`` and ``y`` is introduced. In the example below, the arrays ``x`` and ``y``
    will be assumed to store the halo and galaxy properties in a particular bin of the
    primary halo and galaxy property. To fully implement CAM, the user will need to do
    their own binning as appropriate to the particular problem.

    >>> num_halos_in_mpeak_bin = int(1e4)
    >>> spin_at_fixed_mpeak = 10**np.random.normal(loc=-1.5, size=num_halos_in_mpeak_bin, scale=0.3)

    >>> num_gals_in_mstar_bin = int(1e3)
    >>> some_galprop_at_fixed_mpeak = np.random.power(2.5, size=num_gals_in_mstar_bin)
    >>> modeled_galprop = conditional_abunmatch(spin_at_fixed_mpeak, some_galprop_at_fixed_mpeak)

    Notes
    -----
    In principle, this function is also applicable to traditional abundance matching, e.g.,
    between stellar mass and halo. In practice, this may not be suitable for your application
    if extrapolation beyond the tabulated SMF is needed - the implementation here makes a simple
    call to `numpy.interp`, which can result in undesired edge case behavior if
    a large fraction of model galaxies lie outside the range of the data.

    For a code that provides careful treatment of this extrapolation
    for Schechter-like abundance functions, see the
    `deconvolution abundance matching code <https://bitbucket.org/yymao/abundancematching/>`_
    written by Yao-Yuan Mao.
    """
    x_percentiles = its.rank_order_percentile(x)
    noisy_x_percentile = randomly_resort(x_percentiles, sigma, seed=seed)

    x_table, y_table = its.build_cdf_lookup(y, npts_lookup_table)

    return its.monte_carlo_from_cdf_lookup(x_table, y_table,
            mc_input=noisy_x_percentile)


def randomly_resort(x, sigma, seed=None):
    """ Function randomizes the entries of ``x``
    with an input level of stochasticity ``sigma``.

    Parameters
    -----------
    x : ndarray
        Input array of shape (npts, ) that will be randomly reordered

    sigma : float
        Input level of stochasticity in the randomization

    seed : int, optional
        Seed used to randomly add noise

    Returns
    -------
    noisy_x : ndarray
        Array of shape (npts, )
    """
    npts = len(x)
    noisy_indices = noisy_indexing_array(npts, sigma, seed=seed)

    idx_sorted = np.argsort(x)
    x_sorted = x[idx_sorted]

    noisy_x_sorted = x_sorted[noisy_indices]

    idx_unsorted = unsorting_indices(idx_sorted)

    return noisy_x_sorted[idx_unsorted]


def noisy_indexing_array(npts, sigma, seed=None):
    """
    Function calculates an indexing array that can be used to randomly reorder
    elements of a sorted array. The level of stochasticity in this random reordering
    is set by the input ``sigma``.

    Parameters
    ----------
    npts : int
        Number of points in the sample

    sigma : float or ndarray
        Level of Gaussian noise to add to the rank-ordering.
        When passing a float, noise will be constant. Otherwise,
        must pass an array of shape (npts, ).

    seed : int, optional
        Seed used to randomly draw from a Gaussian

    Returns
    --------
    indices_with_noise : ndarray
        Numpy integer array of shape (npts, ) storing the integers
        0, 1, 2, ..., npts-1 in a noisy order
    """
    sigma = np.atleast_1d(sigma)
    if len(sigma) == 1:
        sigma = np.zeros(npts) + sigma[0]

    if np.any(sigma) < 0:
        msg = "Input ``sigma`` must be non-negative"
        raise ValueError(msg)
    elif np.all(sigma) == 0:
        return np.arange(npts)
    else:
        sigma = np.where(sigma < 1e-3, 1e-3, sigma)

    with NumpyRNGContext(seed):
        noise = np.random.normal(loc=0, scale=sigma, size=npts)

    sorted_ranks = np.arange(1, npts+1)
    sorted_percentiles = sorted_ranks/float(npts+1)
    noisy_percentiles = sorted_percentiles + noise

    # rescale to the unit interval
    noisy_percentiles -= np.min(noisy_percentiles)
    noisy_percentiles /= np.max(noisy_percentiles)

    # Now transform the noisy percentile into an array of (noisy) indices
    noisy_indices = np.array(noisy_percentiles*npts).astype(int)

    # Return a set of noisy percentile values
    # that will be linearly spaced in the unit interval
    # and sorted according to the noisy_index calculated above
    placeholder_negatives = np.zeros_like(noisy_indices) - 1.
    rescaled_noisy_percentile = np.insert(placeholder_negatives,
        noisy_indices, sorted_ranks-1)
    mask_out_placeholder_negatives = rescaled_noisy_percentile!=-1
    indices_with_noise = rescaled_noisy_percentile[mask_out_placeholder_negatives]
    return indices_with_noise.astype(int)
