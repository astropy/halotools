""" Module storing the Numpy kernel for conditional abundance matching
"""
import numpy as np
from astropy.utils import NumpyRNGContext
from ...utils import inverse_transformation_sampling as its
from ...utils import unsorting_indices


__author__ = ('Andrew Hearin', 'Duncan Campbell')
__all__ = ('conditional_abunmatch', 'randomly_resort')


def conditional_abunmatch(haloprop, galprop, sigma=0., npts_lookup_table=1000, seed=None):
    """ Function used to model a correlation between two variables,
     ``haloprop`` and ``galprop``, using conditional abundance matching.

    Parameters
    -----------
    haloprop : ndarray
        Numpy array of shape (num_halos, ) typically storing a halo property

    galprop : ndarray
        Numpy array of shape (num_gals, ) typically storing a galaxy property

    sigma : float, optional
        Level of Gaussian noise that will be introduced
        to the haloprop--galprop correlation.

        Default is 0, for a perfect monotonic relation between haloprop and galprop.

    npts_lookup_table : int, optional
        Size of the lookup table used to approximate the ``galprop`` distribution.

        Default is 1000.

    seed : int, optional
        Random number seed used to introduce noise in the haloprop--galprop correlation.

        Default is None for stochastic results.

    Returns
    -------
    model_galprop : ndarray
        Numpy array of shape (num_halos, ) storing the modeled galprop-values associated
        with each value of the input ``haloprop``.

    Examples
    --------
    Suppose we would like to do some CAM-style modeling of a correlation between some
    halo property ``haloprop`` and some galaxy property ``galprop``.
    The `conditional_abunmatch` function
    can be used to map values of the galaxy property onto the halos in such a way that the
    PDF of ``galprop`` is preserved and a correlation (of variable strength)
    between ``haloprop`` and ``galprop`` is introduced. In the example below,
    the arrays ``haloprop`` and ``galprop``
    will be assumed to store the halo and galaxy properties in a particular bin of the
    primary halo and galaxy property. To fully implement CAM, the user will need to do
    their own binning as appropriate to the particular problem.

    >>> num_halos_in_mpeak_bin = int(1e4)
    >>> mean, size, std = -1.5, num_halos_in_mpeak_bin, 0.3
    >>> spin_at_fixed_mpeak = 10**np.random.normal(loc=mean, size=size, scale=std)
    >>> num_gals_in_mstar_bin = int(1e3)
    >>> some_galprop_at_fixed_mstar = np.random.power(2.5, size=num_gals_in_mstar_bin)
    >>> modeled_galprop = conditional_abunmatch(spin_at_fixed_mpeak, some_galprop_at_fixed_mstar)

    Notes
    -----
    To approximate the input ``galprop`` distribution, the implementation of `conditional_abunmatch`
    builds a lookup table for the CDF of the input ``galprop``  using a simple call to `numpy.interp`,
    which can result in undesired edge case behavior if
    a large fraction of model galaxies lie outside the range of the data.
    To ensure your results are not impacted by this, make sure that
    num_gals >> npts_lookup_table.

    For code that provides careful treatment of this extrapolation
    for traditional abundance matching applications involving Schechter-like abundance functions, see the
    `deconvolution abundance matching code <https://bitbucket.org/yymao/abundancematching/>`_
    written by Yao-Yuan Mao.

    The intended applications of the `conditional_abunmatch` function are those for which
    the stochasticity in ``galprop`` at fixed ``haloprop`` need not be forced
    to respect the form of a log-normal distribution,
    e.g., in models analogous to `age matching <https://arxiv.org/abs/1304.5557/>`_.
    """
    haloprop_table, galprop_table = its.build_cdf_lookup(galprop, npts_lookup_table)
    haloprop_percentiles = its.rank_order_percentile(haloprop)
    noisy_haloprop_percentiles = randomly_resort(haloprop_percentiles, sigma, seed=seed)
    return its.monte_carlo_from_cdf_lookup(haloprop_table, galprop_table,
            mc_input=noisy_haloprop_percentiles)


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
    idx_sorted = np.argsort(x)
    x_sorted = x[idx_sorted]

    noisy_indices = noisy_indexing_array(npts, sigma, seed=seed)
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
        sigma = np.maximum(sigma, 1e-3)

    with NumpyRNGContext(seed):
        noise = np.random.normal(loc=0, scale=sigma, size=npts)

    sorted_ranks = np.arange(1, npts+1)
    sorted_percentiles = sorted_ranks/float(npts+1)
    noisy_percentiles = sorted_percentiles + noise

    # rescale to the unit interval
    noisy_percentiles -= np.min(noisy_percentiles)
    noisy_percentiles /= np.max(noisy_percentiles)

    # Now transform noisy_percentiles into an array of noisy indices
    noisy_indices = np.array(noisy_percentiles*npts).astype(int)

    # At this point, noisy_indices has the appropriate stochasticity but may have repeated entries
    # Our goal is to return a length-npts array with no repeated entries
    # that may be treated as a fancy indexing array to introduce a noisy ordering
    # of some other length-npts array storing our galaxy property.
    # So what we do next is address the issue of repeated entries,
    # replacing them with their rank-order in sequence of their appearance
    placeholder_negatives = np.zeros_like(noisy_indices) - 1.
    rescaled_noisy_percentile = np.insert(placeholder_negatives, noisy_indices, sorted_ranks-1)
    mask_out_placeholder_negatives = rescaled_noisy_percentile != -1
    indices_with_noise = rescaled_noisy_percentile[mask_out_placeholder_negatives]
    return indices_with_noise.astype(int)
