"""
"""
import numpy as np

from ..halo_mass_quenching import HaloMassInterpolQuenching

__all__ = ('test_boundary_values', 'test_extrapolation', 'test_log_convention')


def test_boundary_values():

    model = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e15], [0.25, 0.9])
    mass_array = np.logspace(12, 15, 1000)
    quiescent_fraction = model.mean_quiescent_fraction(prim_haloprop=mass_array)
    assert quiescent_fraction[0] == 0.25
    assert quiescent_fraction[-1] == 0.9
    assert np.all(quiescent_fraction <= 0.9)
    assert np.all(quiescent_fraction >= 0.25)


def test_extrapolation():

    model = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e15], [0.25, 0.9])
    mass_array = np.logspace(10, 20, 1000)
    quiescent_fraction = model.mean_quiescent_fraction(prim_haloprop=mass_array)

    assert quiescent_fraction[0] == 0.
    assert quiescent_fraction[-1] == 1.
    assert np.all(quiescent_fraction >= 0.)
    assert np.all(quiescent_fraction <= 1.)


def test_log_convention():
    model = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e15], [0.25, 0.9])

    quiescent_fraction = model.mean_quiescent_fraction(prim_haloprop=1e12)
    assert quiescent_fraction == 0.25

    quiescent_fraction = model.mean_quiescent_fraction(prim_haloprop=1e15)
    assert quiescent_fraction == 0.9
