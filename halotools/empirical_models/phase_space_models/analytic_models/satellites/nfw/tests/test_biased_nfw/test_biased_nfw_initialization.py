""" Verify that the BiasedNFWPhaseSpace class has the appropriate attributes
after initialization, before and after lookup tables are built.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace

__all__ = ('test_biased_nfw_phase_space_initialization', )


conc_bins = np.linspace(5, 10, 3)
gal_bias_bins = np.linspace(0.1, 20, 2)
gal_bias_bins = np.insert(gal_bias_bins, np.searchsorted(gal_bias_bins, 1), 1)


def test_constructor1():
    r""" Test that composite phase space models have all the appropriate attributes.
    """
    nfw = BiasedNFWPhaseSpace()

    # NFWPhaseSpace attributes
    assert hasattr(nfw, 'assign_phase_space')
    assert hasattr(nfw, '_galprop_dtypes_to_allocate')

    # AnalyticDensityProf attributes
    assert hasattr(nfw, 'circular_velocity')

    # NFWProfile attributes
    assert hasattr(nfw, 'mass_density')
    assert hasattr(nfw, 'halo_prof_param_keys')
    assert hasattr(nfw, 'gal_prof_param_keys')
    assert hasattr(nfw, '_mc_dimensionless_radial_distance')

    # concentration-mass relation
    assert hasattr(nfw, 'conc_NFWmodel')
    assert hasattr(nfw, 'conc_mass_model')


def test_biased_nfw_phase_space_initialization():
    model = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
            conc_gal_bias_bins=gal_bias_bins)
    correct_seq = ['calculate_conc_gal_bias', 'assign_phase_space']
    assert model._mock_generation_calling_sequence == correct_seq


def test_biased_nfw_phase_space_lookup_tables():
    model = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
            conc_gal_bias_bins=gal_bias_bins)

    # MonteCarloGalProf attributes
    assert not hasattr(model, 'logradius_array')
    assert not hasattr(model, 'rad_prof_func_table')
    assert not hasattr(model, 'vel_prof_func_table')

    model.build_lookup_tables()

    assert hasattr(model, 'logradius_array')
    assert hasattr(model, 'rad_prof_func_table')
    assert hasattr(model, 'vel_prof_func_table')

    assert np.allclose(model._conc_gal_bias_lookup_table_bins, gal_bias_bins)
    assert np.allclose(model._conc_NFWmodel_lookup_table_bins, conc_bins)

    assert hasattr(model, 'rad_prof_func_table')
    npts_conc, npts_conc_bias = len(conc_bins), len(gal_bias_bins)
    assert model.rad_prof_func_table.shape == (npts_conc, npts_conc_bias)


def test_raises_memory_warning():
    cmin, cmax, dc = 0.5, 30., 0.25
    bmin, bmax, db = 0.5, 5., 0.1
    model = BiasedNFWPhaseSpace(concentration_binning=(cmin, cmax, dc),
            conc_gal_bias_binning=(bmin, bmax, db))


def test_biased_nfw_phase_space_param_dict1():
    model = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
            conc_gal_bias_bins=gal_bias_bins)
    assert 'conc_gal_bias' in model.param_dict.keys()
    assert 'conc_gal_bias_param0' not in model.param_dict.keys()


def test_biased_nfw_phase_space_param_dict2():
    model = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins,
            conc_gal_bias_logM_abscissa=[14.5, 16])
    assert 'conc_gal_bias_param0' in model.param_dict.keys()
    assert 'conc_gal_bias_logM_abscissa_param0' in model.param_dict.keys()
    assert 'conc_gal_bias' not in model.param_dict.keys()
