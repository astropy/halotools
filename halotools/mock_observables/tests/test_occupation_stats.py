"""
"""
import numpy as np
import pytest
from ...empirical_models import PrebuiltHodModelFactory
from ...sim_manager import FakeSim
from ..occupation_stats import hod_from_mock

__all__ = ('test_occupation_stats1', )


@pytest.mark.installation_test
def test_occupation_stats1():
    haloprop_galaxies = np.array((1.5, 2.5, 4.5))
    haloprop_halos = np.array((1.5, 2.5, 4.5))
    haloprop_bins = np.arange(6)
    mean_occupation, bin_edges = hod_from_mock(haloprop_galaxies, haloprop_halos, haloprop_bins)
    assert np.shape(bin_edges) == np.shape(haloprop_bins)
    assert len(mean_occupation) == bin_edges.shape[0]-1
    assert np.allclose(mean_occupation, np.array((0, 1, 1, 0, 1)))


def test_occupation_stats2():
    haloprop_galaxies = np.array((1.5, 2.5, 4.5))
    haloprop_halos = np.array((1.5, 1.5, 4.5))
    haloprop_bins = np.arange(6)
    with pytest.raises(ValueError) as err:
        __ = hod_from_mock(haloprop_galaxies, haloprop_halos, haloprop_bins)
    substr = "The bin with edges (2.000, 3.000) has galaxies but no halos."
    assert substr in err.value.args[0]


def test_occupation_stats3():
    haloprop_galaxies = np.array((1.5, 2.5, 4.5))
    haloprop_halos = np.array((1.5, 2.5, 2.5))
    haloprop_bins = np.arange(6)
    with pytest.raises(ValueError) as err:
        __ = hod_from_mock(haloprop_galaxies, haloprop_halos, haloprop_bins)
    substr = "The bin with edges (4.000, 5.000) has galaxies but no halos."
    assert substr in err.value.args[0]


def test_occupation_stats4():
    model = PrebuiltHodModelFactory('zheng07', threshold=-21)
    halocat = FakeSim(num_halos_per_massbin=1000)
    model.populate_mock(halocat, seed=44)
    cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
    satmask = model.mock.galaxy_table['gal_type'] == 'satellites'
    cens = model.mock.galaxy_table[cenmask]
    sats = model.mock.galaxy_table[satmask]
    halos = model.mock.halo_table

    log_masses = np.log10(np.sort(list(set(halos['halo_mvir']))))
    max_logmass = log_masses[-1]
    log_bins = np.append(log_masses - 0.1, max_logmass + 0.1)
    haloprop_bins = 10**log_bins
    haloprop_mids = 10**log_masses

    haloprop_halos = halos['halo_mvir']

    # Centrals calculation
    haloprop_galaxies = cens['halo_mvir']
    mean_ncen, __ = hod_from_mock(haloprop_galaxies, haloprop_halos, haloprop_bins)
    assert np.all(mean_ncen >= 0)
    assert np.all(mean_ncen <= 1)
    expected_mean_ncen = model.mean_occupation_centrals(prim_haloprop=haloprop_mids)
    assert np.allclose(expected_mean_ncen, mean_ncen, atol=0.02)

    # Satellites calculation
    haloprop_galaxies = sats['halo_mvir']
    mean_nsat, __ = hod_from_mock(haloprop_galaxies, haloprop_halos, haloprop_bins)
    assert np.all(mean_nsat >= 0)
    expected_mean_nsat = model.mean_occupation_satellites(prim_haloprop=haloprop_mids)

    mean_nsat = np.where(mean_nsat < 1e-2, 0, mean_nsat)
    expected_mean_nsat = np.where(expected_mean_nsat < 1e-2, 0, mean_nsat)
    assert np.allclose(expected_mean_nsat, mean_nsat, rtol=0.02)
