"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace
from ...nfw_phase_space import NFWPhaseSpace

__all__ = ('test_biased_unbiased_agreement1', )


conc_bins = np.linspace(2, 30, 10)
gal_bias_bins = np.linspace(0.1, 20, 5)
gal_bias_bins = np.insert(gal_bias_bins, np.searchsorted(gal_bias_bins, 1), 1)


def test_biased_unbiased_agreement1():
    """
    """
    nfw = NFWPhaseSpace()
    biased_nfw = BiasedNFWPhaseSpace()

    rbins = np.logspace(-2, 0, 100)
    c, b = 5, 1
    dimless_vrad1 = nfw.dimensionless_radial_velocity_dispersion(rbins, c)
    dimless_vrad2 = biased_nfw.dimensionless_radial_velocity_dispersion(rbins, c, b)
    assert np.allclose(dimless_vrad1, dimless_vrad2)


def test_biased_unbiased_disagreement1():
    """
    """
    nfw = NFWPhaseSpace()
    biased_nfw = BiasedNFWPhaseSpace()

    rbins = np.logspace(-2, 0, 100)
    c, b = 5, 2
    dimless_vrad1 = nfw.dimensionless_radial_velocity_dispersion(rbins, c)
    dimless_vrad2 = biased_nfw.dimensionless_radial_velocity_dispersion(rbins, c, b)
    assert not np.allclose(dimless_vrad1, dimless_vrad2)
