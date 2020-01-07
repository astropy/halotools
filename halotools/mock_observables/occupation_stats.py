""" Module storing functions to calculate common halo occupation statistics
such as the first occupation moment.
"""
import numpy as np
from scipy.stats import binned_statistic
from ..utils import crossmatch


__all__ = ('hod_from_mock', 'get_haloprop_of_galaxies')
__author__ = ('Andrew Hearin', )


def hod_from_mock(haloprop_galaxies, haloprop_halos, haloprop_bins=None):
    r"""
    Calculate the HOD of a mock galaxy sample. It returns the expected number
    of galaxies per halo, in bins of whatever halo property
    ``haloprop_galaxies`` and ``haloprop_halos`` are given in.

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

        Note that this property (e.g. halo mass) must be the same as used for
        ``haloprop_halos``.

    haloprop_bins : ndarray, optional
        Array defining the bin edges. If None, this defaults to 10 linearly
        spaced bins and so you will probably obtain better results if you
        pass in logarithmic quantities for the ``haloprop_galaxies``
        and ``haloprop_halos`` arrays.

    Returns
    -------
    mean_occupation : ndarray
        Array of shape (num_bins-1, ) storing the mean occupation
        of the input galaxy sample as a function of the input halo property.

    bin_edges : ndarray
        Array of shape (num_bins, ) storing the bin edges used in the calculation.

    Examples
    --------
    In the following calculation, we'll populate a mock catalog and then manually
    compute the central galaxy HOD (number of central galaxies above the mass
    threshold as a function of halo mass) from the ``galaxy_table``.

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> from halotools.sim_manager import FakeSim
    >>> from halotools.mock_observables import hod_from_mock
    >>> model = PrebuiltHodModelFactory('leauthaud11', threshold=10.75)
    >>> halocat = FakeSim()
    >>> model.populate_mock(halocat)

    Now compute :math:`\langle N_{\rm cen} \rangle(M_{\rm vir})`:

    >>> cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
    >>> central_host_mass = model.mock.galaxy_table['halo_mvir'][cenmask]
    >>> halo_mass = model.mock.halo_table['halo_mvir']
    >>> haloprop_bins = np.logspace(11, 15, 15)
    >>> mean_ncen, bin_edges = hod_from_mock(central_host_mass, halo_mass, haloprop_bins)
    """
    _result = binned_statistic(haloprop_galaxies, haloprop_galaxies, bins=haloprop_bins, statistic='count')
    galaxy_counts, bin_edges, bin_number = _result
    _result = binned_statistic(haloprop_halos, haloprop_halos, bins=haloprop_bins, statistic='count')
    halo_counts, bin_edges, bin_number = _result

    bins_with_galaxies_but_no_halos = (galaxy_counts > 0) & (halo_counts == 0)
    if np.any(bins_with_galaxies_but_no_halos):
        bad_bin_index = np.where(bins_with_galaxies_but_no_halos == True)[0][0]
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
    """ Determine the halo property in ``haloprop_halos`` for each galaxy.
    This crossmatches the galaxy catalog with the halo catalog using their
    ``halo_id``. Return the halo property for galaxies with a match, else nan.

    Parameters
    ----------
    halo_id_galaxies : ndarray
        Integer array of shape (num_galaxies, ) storing the ``halo_id``
        that each galaxy belongs to.

    halo_id_halos : ndarray
        Integer array of shape (num_halos, ) storing the ``halo_id``
        of every host halo in the entire halo catalog used to populate the mock.
        Repeated entries are not permissible,
        but halos with zero or multiple galaxies are accepted.

    haloprop_halos : ndarray
        Array of shape (num_halos, ) storing the halo property of interest,
        e.g., ``halo_vpeak`` or ``halo_spin``.

    Returns
    -------
    haloprop_galaxies : ndarray
        Array of shape (num_galaxies, ) storing the property of the halo
        that each galaxy belongs to. Galaxies with no matching halo will
        receive value of `~numpy.nan`

    Examples
    --------
    When you populate a mock catalog, the host halo mass of every galaxy is automatically
    included in the ``galaxy_table``. However, you may wish to know other halo properties
    for each mock galaxy, such as the spin of the halo the galaxy lives in. The code below
    demonstrates how to use the `get_haloprop_of_galaxies` function to do this.

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> from halotools.sim_manager import FakeSim
    >>> from halotools.mock_observables import get_haloprop_of_galaxies
    >>> model = PrebuiltHodModelFactory('leauthaud11')
    >>> halocat = FakeSim()
    >>> model.populate_mock(halocat)
    >>> halo_id_halos = halocat.halo_table['halo_id']
    >>> halo_id_galaxies = model.mock.galaxy_table['halo_id']
    >>> haloprop_halos = halocat.halo_table['halo_spin']
    >>> halo_spin_galaxies = get_haloprop_of_galaxies(halo_id_galaxies, halo_id_halos, haloprop_halos)
    >>> model.mock.galaxy_table['halo_spin'] = halo_spin_galaxies

    Note that we needed to use the original halo catalog
    to retrieve the ``halo_spin`` of the halos; in order to save memory,
    the version of the ``halo_table`` that is bound to ``model.mock`` has a
    restricted subset of columns.
    """
    halo_id_galaxies = np.atleast_1d(halo_id_galaxies)
    halo_id_halos = np.atleast_1d(halo_id_halos)
    haloprop_halos = np.atleast_1d(haloprop_halos)
    result = np.zeros_like(halo_id_galaxies) + np.nan
    idxA, idxB = crossmatch(halo_id_galaxies, halo_id_halos)
    result[idxA] = haloprop_halos[idxB]
    return result
