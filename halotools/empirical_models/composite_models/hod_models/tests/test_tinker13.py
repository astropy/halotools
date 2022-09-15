"""
"""
import numpy as np
import pytest

from ....factories import PrebuiltHodModelFactory

from .....sim_manager import FakeSim


__all__ = ("test_tinker13_instantiation", "test_tinker13_populate")


def test_tinker13_instantiation():
    """Verify that we can successfully instantiate the tinker13 model"""
    model = PrebuiltHodModelFactory("tinker13")

    model = PrebuiltHodModelFactory(
        "tinker13",
        quiescent_fraction_abscissa=[1e12, 1e13, 1e14, 1e15],
        quiescent_fraction_ordinates=[0.25, 0.5, 0.75, 0.9],
    )


def test_tinker13_populate():
    """Demonstrate that Tinker13 model successfully populates, and also that
    the monte carlo realization has a statistically consistent quiescent fraction
    with the underlying model.

    Note that this serves as a regression test for https://github.com/astropy/halotools/issues/672.
    """
    model = PrebuiltHodModelFactory(
        "tinker13",
        quiescent_fraction_abscissa=[1e12, 1e15],
        quiescent_fraction_ordinates=[0.5, 0.5],
    )
    fake_sim = FakeSim(num_halos_per_massbin=500)
    model.populate_mock(fake_sim, seed=43)

    mask = model.mock.galaxy_table["halo_mvir"] > 1e12
    mask *= model.mock.galaxy_table["halo_mvir"] < 1e15
    mask *= model.mock.galaxy_table["gal_type"] == "centrals"
    cens = model.mock.galaxy_table[mask]
    mc_quiescent_fraction = np.mean(cens["central_sfr_designation"] == "quiescent")
    assert np.allclose(mc_quiescent_fraction, 0.5, rtol=0.1)
