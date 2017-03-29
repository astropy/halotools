"""
"""
import numpy as np
from astropy.utils.data import get_pkg_data_filename

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


def test_unbiased_vel_rad_disp_external_ch10_cg5():
    halo_conc, gal_conc = 10, 5
    fname = get_pkg_data_filename('data/van_den_bosch_nfw_vr_disp_ch10_cg5.dat')
    x = np.loadtxt(fname)
    frank_r_by_Rvir = x[:, 1]
    frank_dimless_sigma_rad = x[:, 2]
    aph_result = biased_dimless_vel_rad_disp(frank_r_by_Rvir, halo_conc, gal_conc)
    assert np.allclose(aph_result, frank_dimless_sigma_rad, rtol=1e-3)


def test_unbiased_vel_rad_disp_external_ch5_cg10():
    halo_conc, gal_conc = 5, 10
    fname = get_pkg_data_filename('data/van_den_bosch_nfw_vr_disp_ch5_cg10.dat')
    x = np.loadtxt(fname)
    frank_r_by_Rvir = x[:, 1]
    frank_dimless_sigma_rad = x[:, 2]
    aph_result = biased_dimless_vel_rad_disp(frank_r_by_Rvir, halo_conc, gal_conc)
    assert np.allclose(aph_result, frank_dimless_sigma_rad, rtol=1e-3)
