"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pytest

from astropy.cosmology import WMAP9, Planck13, Planck15
from astropy import units as u

from ..halo_boundary_functions import density_threshold, delta_vir
from ..halo_boundary_functions import halo_radius_to_halo_mass, halo_mass_to_halo_radius

from .....custom_exceptions import HalotoolsError

__all__ = ("test_halo_radius_to_halo_mass", "test_delta_vir")


def test_halo_radius_to_halo_mass():
    r"""Check that the `~halotools.empirical_models.halo_boundary_functions.halo_mass_to_halo_radius`
    and  `~halotools.empirical_models.halo_boundary_functions.halo_radius_to_halo_mass` functions are
    proper inverses of one another for a range of mdef, cosmology, and redshift
    """
    r0 = 0.25

    for cosmology in (WMAP9, Planck13):
        for redshift in (0, 1, 5, 10):
            for mdef in ("vir", "200m", "2500c"):
                m1 = halo_radius_to_halo_mass(r0, cosmology, redshift, mdef)
                r1 = halo_mass_to_halo_radius(m1, cosmology, redshift, mdef)
                assert np.allclose(r1, r0, rtol=1e-3)


def test_delta_vir():
    r"""Compute the calculated value of `~halotools.empirical_models.halo_boundary_functions.delta_vir`
    at high-redshift where :math:`\Omega_{\rm m} = 1` should be a good approximation, and
    compare it to the analytical top-hat collapse result in this regime.
    """
    bn98_result = delta_vir(WMAP9, 10.0)
    assert np.allclose(bn98_result, 18.0 * np.pi**2, rtol=0.01)

    # Choose a high-redshift where Om = 1 is a good approximation
    z = 10
    rho_crit = WMAP9.critical_density(z)
    rho_crit = rho_crit.to(u.Msun / u.Mpc**3).value / WMAP9.h**2
    rho_m = WMAP9.Om(z) * rho_crit
    wmap9_delta_vir_z10 = density_threshold(WMAP9, z, "vir") / rho_m

    assert np.allclose(wmap9_delta_vir_z10, bn98_result, rtol=0.01)


def test_density_threshold():
    r"""Verify that the `~halotools.empirical_models.halo_boundary_functions.density_threshold`
    method returns the correct multiple of the appropriate density contrast over a range
    of redshifts and cosmologies.

    """

    zlist = [0, 1, 5, 10]
    for z in zlist:
        for cosmo in (WMAP9, Planck13):
            rho_crit = WMAP9.critical_density(z)
            rho_crit = rho_crit.to(u.Msun / u.Mpc**3).value / WMAP9.h**2
            rho_m = WMAP9.Om(z) * rho_crit

            wmap9_200c = density_threshold(WMAP9, z, "200c") / rho_crit
            assert np.allclose(wmap9_200c, 200.0, rtol=0.01)

            wmap9_2500c = density_threshold(WMAP9, z, "2500c") / rho_crit
            assert np.allclose(wmap9_2500c, 2500.0, rtol=0.01)

            wmap9_200m = density_threshold(WMAP9, z, "200m") / rho_m
            assert np.allclose(wmap9_200m, 200.0, rtol=0.01)


def test_density_threshold_error_handling():
    r"""Verify that we raise a `~halotools.custom_exceptions.HalotoolsError` when nonsense
    inputs such as 'Jose Canseco' are passed to the
    `~halotools.empirical_models.halo_boundary_functions.density_threshold` method.
    """

    with pytest.raises(HalotoolsError):
        result = density_threshold(WMAP9, 0.0, "Jose Canseco")

    with pytest.raises(HalotoolsError):
        result = density_threshold(WMAP9, 0.0, "250.m")

    with pytest.raises(HalotoolsError):
        result = density_threshold(WMAP9, 0.0, "250b")

    with pytest.raises(HalotoolsError):
        result = density_threshold(WMAP9, 0.0, "-250m")

    with pytest.raises(HalotoolsError):
        result = density_threshold("Jose Canseco", 0.0, "vir")


def test_halo_mass_to_halo_radius_colossus_consistency():
    """ """
    mass_array = np.logspace(10, 15, 10)
    halotools_radius_z0 = halo_mass_to_halo_radius(mass_array, Planck15, 0.0, "vir")
    halotools_radius_z1 = halo_mass_to_halo_radius(mass_array, Planck15, 1.0, "vir")
    try:
        from colossus.halo.mass_so import M_to_R
        from colossus.cosmology import cosmology

        cosmo = cosmology.setCosmology("planck15")
        colossus_radius_z0 = M_to_R(mass_array, 0.0, "vir") / 1000.0
        colossus_radius_z1 = M_to_R(mass_array, 1.0, "vir") / 1000.0
        assert np.allclose(colossus_radius_z0, halotools_radius_z0, rtol=0.01)
        assert np.allclose(colossus_radius_z1, halotools_radius_z1, rtol=0.01)
    except ImportError:
        pass
