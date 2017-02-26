""" Module storing functions to calculate common halo occupation statistics
such as the first occupation moment.
"""
import numpy as np
from scipy.stats import binned_statistic
from ..utils import crossmatch


__all__ = ('hod_from_mock',)
__author__ = ('Andrew Hearin', )


def hod_from_mock(haloprop_galaxies, haloprop_halos, haloprop_bins=None):
    """
    Calculate the HOD of a mock galaxy sample.

    Parameters
    ----------
    haloprop_galaxies : ndarray
        Array of shape (num_galaxies, ) used to bin the galaxies
        according to the property of their host halo, e.g., host halo mass.

        If this quantity is not readily available in the mock galaxy catalog,
        the `get_haloprop_of_galaxies` function can be used to calculate it.

    haloprop_halos : ndarray
        Array of shape (num_halos, ) used to bin the halos in the same manner
        as the galaxies so that the counts in each bin can be properly normalized.

    haloprop_bins : ndarray, optional
        Array defining the bin edges. If this array is not passed, then you will probably
        obtain better results if you pass in logarithmic quantities for the
        ``haloprop_galaxies`` and ``haloprop_halos`` arrays.

    Returns
    -------
    mean_occupation : ndarray
        Array of shape (num_bins-1, ) storing the mean occupation
        of the input galaxy sample as a function of the input halo property.

    bin_edges : ndarray
        Array of shape (num_bins, ) storing the bin edges used in the calculation.

    Examples
    --------
    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> from halotools.sim_manager import FakeSim
    >>> model = PrebuiltHodModelFactory('leauthaud11', threshold=10.75)
    >>> halocat = FakeSim()
    >>> model.populate_mock(halocat)
    >>> cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
    >>> central_host_mass = model.mock.galaxy_table['halo_mvir'][cenmask]
    >>> halo_mass = model.mock.halo_table['halo_mvir']
    >>> haloprop_bins = np.logspace(11, 15, 15)
    >>> mean_ncen, bin_edges = hod_from_mock(central_host_mass, halo_mass, haloprop_bins)
    """
    _result = binned_statistic(haloprop_galaxies, None, bins=haloprop_bins, statistic='count')
    galaxy_counts, bin_edges, bin_number = _result
    _result = binned_statistic(haloprop_halos, None, bins=haloprop_bins, statistic='count')
    halo_counts, bin_edges, bin_number = _result

    bins_with_galaxies_but_no_halos = (galaxy_counts > 0) & (halo_counts == 0)
    if np.any(bins_with_galaxies_but_no_halos):
        bad_bin_index = np.where(bins_with_galaxies_but_no_halos==True)[0][0]
        bad_bin_edge_low, bad_bin_edge_high = bin_edges[bad_bin_index], bin_edges[bad_bin_index+1]
        msg = ("The bin with edges ({0:.3f}, {1:.3f}) has galaxies but no halos.\n"
            "This must mean that the input ``haloprop_galaxies`` and ``haloprop_halos`` arrays \n"
            "have not been consistently computed.\n".format(bad_bin_edge_low, bad_bin_edge_high))
        raise ValueError(msg)

    mean_occupation = np.zeros(len(haloprop_bins)-1)
    halo_mask = halo_counts > 0
    mean_occupation[halo_mask] = galaxy_counts[halo_mask]/halo_counts[halo_mask].astype('f4')
    return mean_occupation, bin_edges


def get_haloprop_of_galaxies(halo_id_galaxies, halo_id_halos, haloprop_halos):
    """ Calculate the host halo property of every galaxy with a ``halo_id`` that
    matches one of input halos. This function can be used, for example,
    to calculate the host halo mass of a galaxy.

    Parameters
    ----------
    halo_id_galaxies : ndarray
        Integer array of shape (num_galaxies, ) storing the ``halo_id``
        that each galaxy belongs to.

    halo_id_halos : ndarray
        Integer array of shape (num_halos, ) storing the ``halo_id``
        of each viable host halo. Repeated entries are not permissible,
        but halos with zero or multiple galaxies are accepted.

    haloprop_halos : ndarray
        Array of shape (num_halos, ) storing the halo property of interest.

    Returns
    -------
    haloprop_galaxies : ndarray
        Array of shape (num_galaxies, ) storing the property of the halo
        that each galaxy belongs to. Galaxies with no matching halo will
        receive value of np.nan

    Examples
    --------
    >>> num_galaxies, num_halos = 10, 100
    >>> halo_id_halos = np.arange(num_halos).astype(int)
    >>> haloprop_halos = np.random.rand(num_halos)
    >>> halo_id_galaxies = np.random.randint(0, num_halos, num_galaxies)
    >>> haloprop_galaxies = get_haloprop_of_galaxies(halo_id_galaxies, halo_id_halos, haloprop_halos)
    """
    halo_id_galaxies = np.atleast_1d(halo_id_galaxies)
    halo_id_halos = np.atleast_1d(halo_id_halos)
    haloprop_halos = np.atleast_1d(haloprop_halos)
    result = np.zeros_like(halo_id_galaxies) + np.nan
    idxA, idxB = crossmatch(halo_id_galaxies, halo_id_halos)
    result[idxA] = haloprop_halos[idxB]
    return result
