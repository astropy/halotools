"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace
from .......factories import PrebuiltHodModelFactory, HodModelFactory
from ........sim_manager import FakeSim
from ........mock_observables import relative_positions_and_velocities

__all__ = ('test_mockpop1', )


conc_bins = np.linspace(2, 30, 3)
gal_bias_bins = np.linspace(0.1, 20, 2)
gal_bias_bins = np.insert(gal_bias_bins, np.searchsorted(gal_bias_bins, 1), 1)


def _nearest_lower_value_mask(arr, bound, epsilon=0.001):

    unique_masses = np.unique(arr)
    idx = np.searchsorted(unique_masses, bound) - 1
    mask_value = unique_masses[idx]
    mask = arr > mask_value*(1 - epsilon)
    mask *= arr < mask_value*(1 + epsilon)
    return mask


def _radial_velocities(galaxy_sample, Lbox):
    galx, hostx = galaxy_sample['x'], galaxy_sample['halo_x']
    galvx, hostvx = galaxy_sample['vx'], galaxy_sample['halo_vx']
    xrel, vxrel = relative_positions_and_velocities(galx, hostx, v1=galvx, v2=hostvx, period=Lbox[0])
    galy, hosty = galaxy_sample['y'], galaxy_sample['halo_y']
    galvy, hostvy = galaxy_sample['vy'], galaxy_sample['halo_vy']
    yrel, vyrel = relative_positions_and_velocities(galy, hosty, v1=galvy, v2=hostvy, period=Lbox[1])
    galz, hostz = galaxy_sample['z'], galaxy_sample['halo_z']
    galvz, hostvz = galaxy_sample['vz'], galaxy_sample['halo_vz']
    zrel, vzrel = relative_positions_and_velocities(galz, hostz, v1=galvz, v2=hostvz, period=Lbox[2])
    r = np.sqrt(xrel**2 + yrel**2 + zrel**2)
    vr = np.zeros_like(r)
    mask = r > 0
    vr[mask] = (xrel[mask]*vxrel[mask] + yrel[mask]*vyrel[mask] + zrel[mask]*vzrel[mask])/r[mask]
    return vr


def enforce_correct_conc_gal_bias(galaxy_sample, correct_conc_gal_bias_value):
    satmask = galaxy_sample['gal_type'] == 'satellites'
    satmsg = ("``galaxy_sample`` has satellites with `conc_gal_bias` that disagrees with {0:.3f}"
        "".format(correct_conc_gal_bias_value))
    assert np.all(galaxy_sample['conc_gal_bias'][satmask] == correct_conc_gal_bias_value), satmsg
    cenmsg = ("``galaxy_sample`` has centrals with non-zero `conc_gal_bias`")
    assert np.all(galaxy_sample['conc_gal_bias'][~satmask] == 0.), cenmsg


def enforce_host_centric_distance_exists(galaxy_sample):
    msg = "host_centric_distance key was never mapped during mock population"
    assert np.any(galaxy_sample['host_centric_distance'] > 0), msg


def mean_satellite_host_centric_dimensionless_distance(galaxy_sample):
    satmask = galaxy_sample['gal_type'] == 'satellites'
    r_by_rvir = galaxy_sample['host_centric_distance'][satmask]/galaxy_sample['halo_rvir'][satmask]
    return np.mean(r_by_rvir)


def test_mockpop1():
    zheng07_model = PrebuiltHodModelFactory('zheng07', threshold=-20.5)
    model_dict = zheng07_model.model_dictionary

    biased_nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins)

    model_dict['satellites_profile'] = biased_nfw
    model = HodModelFactory(**model_dict)
    model.param_dict['conc_gal_bias'] = gal_bias_bins.min()

    halocat = FakeSim(seed=43, num_halos_per_massbin=25)
    model.populate_mock(halocat, seed=44)
    assert 'conc_gal_bias_param0' not in list(model.mock.galaxy_table.keys())

    enforce_correct_conc_gal_bias(model.mock.galaxy_table, gal_bias_bins.min())
    enforce_host_centric_distance_exists(model.mock.galaxy_table)

    r_by_rvir_0p1 = mean_satellite_host_centric_dimensionless_distance(model.mock.galaxy_table)

    satmask_0p1 = model.mock.galaxy_table['gal_type'] == 'satellites'
    sats_0p1 = model.mock.galaxy_table[satmask_0p1]
    min_host_mass_to_search_for = 10**16.001
    mask16_0p1 = _nearest_lower_value_mask(sats_0p1['halo_mvir'],
                        min_host_mass_to_search_for)
    galaxy_sample_m16_0p1 = sats_0p1[mask16_0p1]
    std_vr_0p1 = np.std(_radial_velocities(galaxy_sample_m16_0p1, halocat.Lbox))

    model.param_dict['conc_gal_bias'] = gal_bias_bins.max()
    model.mock.populate(seed=45)
    r_by_rvir_10 = mean_satellite_host_centric_dimensionless_distance(model.mock.galaxy_table)

    msg = "mean host_centric_distance should decrease when increasing ``conc_gal_bias`` param"
    assert r_by_rvir_10 < r_by_rvir_0p1 - 0.1, msg

    satmask_10 = model.mock.galaxy_table['gal_type'] == 'satellites'
    sats_10 = model.mock.galaxy_table[satmask_10]
    min_host_mass_to_search_for = 10**16.001
    mask16_10 = _nearest_lower_value_mask(sats_10['halo_mvir'],
                        min_host_mass_to_search_for)
    galaxy_sample_m16_10 = sats_10[mask16_10]
    std_vr_10 = np.std(_radial_velocities(galaxy_sample_m16_10, halocat.Lbox))

    msg = ("Radial velocity dispersions should be smaller for larger values of ``conc_gal_bias`` ")
    assert std_vr_10 < 1.25*std_vr_0p1, msg


def test_mockpop2():
    zheng07_model = PrebuiltHodModelFactory('zheng07', threshold=-20.5)
    model_dict = zheng07_model.model_dictionary

    conc_gal_bias_logM_abscissa = 14, 16
    biased_nfw = BiasedNFWPhaseSpace(
            concentration_bins=conc_bins, conc_gal_bias_bins=gal_bias_bins,
            conc_gal_bias_logM_abscissa=conc_gal_bias_logM_abscissa,
            conc_mass_model='dutton_maccio14')

    model_dict['satellites_profile'] = biased_nfw
    model = HodModelFactory(**model_dict)
    assert model.param_dict['conc_gal_bias_logM_abscissa_param0'] == 14
    assert model.param_dict['conc_gal_bias_logM_abscissa_param1'] == 16

    model.param_dict['conc_gal_bias_param0'] = gal_bias_bins.min()
    model.param_dict['conc_gal_bias_param1'] = gal_bias_bins.max()

    halocat = FakeSim(seed=43, num_halos_per_massbin=25)
    model.populate_mock(halocat, seed=43)
    assert 'conc_gal_bias' in list(model.mock.galaxy_table.keys())
    assert 'conc_gal_bias_param0' not in list(model.mock.galaxy_table.keys())

    # enforce_correct_conc_gal_bias(model.mock.galaxy_table, 0.1)
    enforce_host_centric_distance_exists(model.mock.galaxy_table)

    # Select a galaxy sample with the same host halo mass ~ 10**14, and another with 10**16
    satmask = model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = model.mock.galaxy_table[satmask]
    min_host_mass_to_search_for = 10**14.001
    mask14 = _nearest_lower_value_mask(sats['halo_mvir'],
                        min_host_mass_to_search_for)
    galaxy_sample_m14 = sats[mask14]
    assert np.all(galaxy_sample_m14['halo_mvir'] == 10**14)
    enforce_correct_conc_gal_bias(galaxy_sample_m14, gal_bias_bins.min())

    min_host_mass_to_search_for = 10**16.001
    mask16 = _nearest_lower_value_mask(sats['halo_mvir'],
                        min_host_mass_to_search_for)
    galaxy_sample_m16 = sats[mask16]
    assert np.all(galaxy_sample_m16['halo_mvir'] == 10**16)
    enforce_correct_conc_gal_bias(galaxy_sample_m16, gal_bias_bins.max())

    r_by_rvir_m14 = mean_satellite_host_centric_dimensionless_distance(galaxy_sample_m14)
    r_by_rvir_m16 = mean_satellite_host_centric_dimensionless_distance(galaxy_sample_m16)
    msg = "mean host_centric_distance should decrease when increasing ``conc_gal_bias`` param"
    assert r_by_rvir_m16 < r_by_rvir_m14 - 0.1, msg

    vr_m16 = _radial_velocities(galaxy_sample_m16, halocat.Lbox)
    inner_mask = galaxy_sample_m16['host_centric_distance']/galaxy_sample_m16['halo_rvir'] < 0.5

    msg = ("Galaxies in inner-half of the halo should have larger radial velocity dispersions \n"
        "relative to galaxies in the outer half of the halo")
    assert np.std(vr_m16[inner_mask]) > 1.25*np.std(vr_m16[~inner_mask]), msg
