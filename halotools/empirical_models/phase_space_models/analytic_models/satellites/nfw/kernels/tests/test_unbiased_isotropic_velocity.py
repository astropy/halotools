"""
"""
import numpy as np
from astropy.utils.data import get_pkg_data_filename

from ..unbiased_isotropic_velocity import dimensionless_radial_velocity_dispersion as unbiased_dimless_vel_rad_disp


__all__ = ('test_unbiased_vel_rad_disp1', )


def test_unbiased_vel_rad_disp1():
    scaled_radius = np.logspace(-2, 0, 25)
    conc = 5
    result = unbiased_dimless_vel_rad_disp(scaled_radius, conc)


def test_unbiased_vel_rad_disp_external_c5():
    fname = get_pkg_data_filename('data/van_den_bosch_nfw_vr_disp_c5.dat')
    x = np.loadtxt(fname)
    frank_r_by_Rvir = x[:, 1]
    frank_dimless_sigma_rad = x[:, 2]
    aph_dimless_sigma_rad = unbiased_dimless_vel_rad_disp(frank_r_by_Rvir, 5)
    assert np.allclose(frank_dimless_sigma_rad, aph_dimless_sigma_rad, rtol=1e-3)


def test_unbiased_vel_rad_disp_external_c10():
    fname = get_pkg_data_filename('data/van_den_bosch_nfw_vr_disp_c10.dat')
    x = np.loadtxt(fname)
    frank_r_by_Rvir = x[:, 1]
    frank_dimless_sigma_rad = x[:, 2]
    aph_dimless_sigma_rad = unbiased_dimless_vel_rad_disp(frank_r_by_Rvir, 10)
    assert np.allclose(frank_dimless_sigma_rad, aph_dimless_sigma_rad, rtol=1e-3)
