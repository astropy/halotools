""" Explicit test that Monte Carlo realizations of BiasedNFWPhaseSpace
do indeed trace an NFW profile.
"""
import numpy as np

from ..test_nfw_profile import analytic_nfw_density_outer_shell_normalization
from ..test_nfw_profile import monte_carlo_density_outer_shell_normalization

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace

__all__ = ['test_mc_dimensionless_radial_distance']

fixed_seed = 43


def test_mc_dimensionless_radial_distance():
    r""" Method used to test `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`.

    Method uses the `~halotools.empirical_models.analytic_nfw_density_outer_shell_normalization` function
    and the `~halotools.empirical_models.monte_carlo_density_outer_shell_normalization` function
    to verify that the points returned by `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`
    do indeed trace an NFW profile.

    """
    conc_bins = np.array((5, 10, 15))
    gal_bias_bins = np.array((1, 2))
    nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
            conc_gal_bias_bins=gal_bias_bins)

    Npts = int(5e5)
    c5 = np.zeros(Npts) + 5
    c10 = np.zeros(Npts) + 10
    c15 = np.zeros(Npts) + 15

    r5 = nfw._mc_dimensionless_radial_distance(c5, 1, seed=43)
    r10 = nfw._mc_dimensionless_radial_distance(c10, 1, seed=43)
    r15 = nfw._mc_dimensionless_radial_distance(c15, 1, seed=43)

    assert np.all(r15 <= 1)
    assert np.all(r15 >= 0)
    assert np.all(r10 <= 1)
    assert np.all(r10 >= 0)
    assert np.all(r5 <= 1)
    assert np.all(r5 >= 0)

    assert np.mean(r15) < np.mean(r10)
    assert np.mean(r10) < np.mean(r5)
    assert np.median(r15) < np.median(r10)
    assert np.median(r10) < np.median(r5)

    num_rbins = 15
    rbins = np.linspace(0.05, 1, num_rbins)
    for r, c in zip([r5, r10, r15], [5, 10, 15]):
        rbin_midpoints, monte_carlo_ratio = (
            monte_carlo_density_outer_shell_normalization(rbins, r))
        analytical_ratio = (
            analytic_nfw_density_outer_shell_normalization(rbin_midpoints, c))
        assert np.allclose(monte_carlo_ratio, analytical_ratio, 0.05)
