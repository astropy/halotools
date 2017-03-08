"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.cosmology import Planck15

from ..surface_density_helpers import annular_area_weighted_midpoints
from ..surface_density_helpers import rho_matter_comoving_in_halotools_units


__all__ = ('test_annular_area_weighted_midpoints_linear_spacing', )

fixed_seed = 43


def test_annular_area_weighted_midpoints_linear_spacing():
    """
    """
    rp_bins = np.linspace(0.5, 10, 25)
    r_equal_area = annular_area_weighted_midpoints(rp_bins)
    assert r_equal_area.shape[0] == rp_bins.shape[0]-1
    inner_area = np.pi*(r_equal_area[0]**2 - rp_bins[0]**2)
    outer_area = np.pi*(rp_bins[1]**2 - r_equal_area[0]**2)
    assert np.allclose(inner_area, outer_area)


def test_annular_area_weighted_midpoints_log_spacing():
    """
    """
    rp_bins = np.logspace(-2, 2, 25)
    r_equal_area = annular_area_weighted_midpoints(rp_bins)
    assert r_equal_area.shape[0] == rp_bins.shape[0]-1
    inner_area = np.pi*(r_equal_area[0]**2 - rp_bins[0]**2)
    outer_area = np.pi*(rp_bins[1]**2 - r_equal_area[0]**2)
    assert np.allclose(inner_area, outer_area)


def test_rho_matter_comoving_in_halotools_units():
    """ Manually verify percent-level agreement for
    rho_matter with hard-coded Planck15 best-fit values
    """
    gee = 4.2994e-9
    rhocrit = 3.E4/(8.*np.pi*gee)
    rhobar = rhocrit*0.3075
    halotools_rhobar = rho_matter_comoving_in_halotools_units(Planck15)
    assert np.allclose(halotools_rhobar, rhobar, rtol=0.01)




