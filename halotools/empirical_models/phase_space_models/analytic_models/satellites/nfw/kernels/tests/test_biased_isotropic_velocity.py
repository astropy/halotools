"""
"""
import numpy as np

from ..unbiased_isotropic_velocity import dimensionless_radial_velocity_dispersion as unbiased_dimless_vel_rad_disp
from ..biased_isotropic_velocity import dimensionless_radial_velocity_dispersion as biased_dimless_vel_rad_disp


__all__ = ('test_unbiased_vel_rad_disp1', )


def test_unbiased_vel_rad_disp1():
    scaled_radius = np.logspace(-2, 0, 25)
    halo_conc = 5
    unbiased_result = unbiased_dimless_vel_rad_disp(scaled_radius, halo_conc)

    gal_conc = halo_conc
    biased_result = biased_dimless_vel_rad_disp(scaled_radius, halo_conc, gal_conc)
    assert np.allclose(unbiased_result, biased_result)


def test_unbiased_vel_rad_disp2():
    scaled_radius = np.logspace(-2, 0, 25)
    halo_conc = 5
    unbiased_result = unbiased_dimless_vel_rad_disp(scaled_radius, halo_conc)

    gal_conc = 2*halo_conc
    biased_result = biased_dimless_vel_rad_disp(scaled_radius, halo_conc, gal_conc)
    assert not np.allclose(unbiased_result, biased_result)
