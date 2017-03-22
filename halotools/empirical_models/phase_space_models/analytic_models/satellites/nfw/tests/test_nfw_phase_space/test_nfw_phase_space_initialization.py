"""
"""
import numpy as np

from ...nfw_phase_space import NFWPhaseSpace


__all__ = ('test_constructor1', )

fixed_seed = 43


def test_constructor1():
    r""" Test that composite phase space models have all the appropriate attributes.
    """
    nfw = NFWPhaseSpace()

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


def test_constructor2():
    r""" Test that composite phase space models have all the appropriate attributes.
    """
    nfw = NFWPhaseSpace(concentration_bins=np.linspace(5, 10, 3))

    # MonteCarloGalProf attributes
    assert not hasattr(nfw, 'logradius_array')
    assert not hasattr(nfw, 'rad_prof_func_table')
    assert not hasattr(nfw, 'vel_prof_func_table')

    nfw.build_lookup_tables()
    assert hasattr(nfw, 'logradius_array')
    assert hasattr(nfw, 'rad_prof_func_table')
    assert hasattr(nfw, 'vel_prof_func_table')
