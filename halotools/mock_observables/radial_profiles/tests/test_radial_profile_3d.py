""" Module providing unit-testing of `~halotools.mock_observables.radial_profile_3d`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..radial_profile_3d import radial_profile_3d
from ...tests.cf_helpers import (
    generate_locus_of_3d_points,
    generate_thin_shell_of_3d_points,
    generate_3d_regular_mesh,
)

import pytest

__all__ = ("test_radial_profile_3d_test1",)

fixed_seed = 44


def test_radial_profile_3d_test1():
    """For a tight localization of sample1 points surrounded by two concentric
    shells of sample2 points, verify that both the counts and the primary result
    of `~halotools.mock_observables.radial_profile_3d` are correct.

    In this test, PBCs are irrelevant.
    """
    npts1 = 100
    xc, yc, zc = 0.5, 0.5, 0.5
    sample1 = generate_locus_of_3d_points(npts1, xc, yc, zc, seed=fixed_seed)

    npts2 = 90
    shell_radii_absolute = np.array([0.01, 0.02, 0.03, 0.04])
    midpoints = (shell_radii_absolute[:-1] + shell_radii_absolute[1:]) / 2.0
    sample2a = generate_thin_shell_of_3d_points(
        npts2, midpoints[0], xc, yc, zc, seed=fixed_seed
    )
    sample2b = generate_thin_shell_of_3d_points(
        npts2, midpoints[1], xc, yc, zc, seed=fixed_seed
    )
    sample2c = generate_thin_shell_of_3d_points(
        npts2, midpoints[2], xc, yc, zc, seed=fixed_seed
    )
    sample2 = np.concatenate([sample2a, sample2b, sample2c])

    a, b, c = 0.5, 1.5, 2.5
    quantity_a = np.zeros(npts2) + a
    quantity_b = np.zeros(npts2) + b
    quantity_c = np.zeros(npts2) + c
    quantity = np.concatenate([quantity_a, quantity_b, quantity_c])

    result, counts = radial_profile_3d(
        sample1,
        sample2,
        quantity,
        rbins_absolute=shell_radii_absolute,
        period=1,
        return_counts=True,
    )
    assert len(result) == len(midpoints)
    assert np.all(result == [a, b, c])
    assert np.all(counts == npts1 * npts2)


def test_radial_profile_3d_test2():
    """For two tight localizations of sample1 points each surrounded by two concentric
    shells of sample2 points, verify that both the counts and the primary result
    of `~halotools.mock_observables.radial_profile_3d` are correct.

    In this test, PBCs have a non-trivial impact on the results.
    """
    npts1a = 100
    xca1, yca1, zca1 = 0.3, 0.3, 0.3
    xca2, yca2, zca2 = 0.9, 0.9, 0.9

    sample1a = generate_locus_of_3d_points(npts1a, xca1, yca1, zca1, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(npts1a, xca2, yca2, zca2, seed=fixed_seed)
    sample1 = np.concatenate([sample1a, sample1b])
    npts1 = len(sample1)

    rbins_absolute = np.array([0.01, 0.03, 0.3])
    shell1_absolute_radius, shell2_absolute_radius = 0.02, 0.2

    sample2_p1_r1 = generate_thin_shell_of_3d_points(
        npts1, shell1_absolute_radius, xca1, yca1, zca1, seed=fixed_seed, Lbox=1
    )
    sample2_p2_r1 = generate_thin_shell_of_3d_points(
        npts1, shell1_absolute_radius, xca2, yca2, zca2, seed=fixed_seed, Lbox=1
    )
    sample2_p1_r2 = generate_thin_shell_of_3d_points(
        npts1, shell2_absolute_radius, xca1, yca1, zca1, seed=fixed_seed, Lbox=1
    )
    sample2_p2_r2 = generate_thin_shell_of_3d_points(
        npts1, shell2_absolute_radius, xca2, yca2, zca2, seed=fixed_seed, Lbox=1
    )
    sample2 = np.concatenate(
        [sample2_p1_r1, sample2_p2_r1, sample2_p1_r2, sample2_p2_r2]
    )
    npts2 = len(sample2)

    a, b = 0.5, 1.5
    quantity_a = np.zeros(int(npts2 / 2)) + a
    quantity_b = np.zeros(int(npts2 / 2)) + b
    quantity = np.concatenate([quantity_a, quantity_b])

    result, counts = radial_profile_3d(
        sample1,
        sample2,
        quantity,
        rbins_absolute=rbins_absolute,
        return_counts=True,
        period=1,
    )
    assert np.all(counts == (npts1 / 2.0) * (npts2 / 2.0))
    assert np.all(result == [0.5, 1.5])


def test_absolute_vs_normalized_agreement():
    npts1, npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        quantity2 = np.random.random(npts2)

    rbins_absolute = np.linspace(0.01, 0.2, 5)
    fixed_rvir = 0.1
    rbins_normalized = rbins_absolute / fixed_rvir

    result1, counts1 = radial_profile_3d(
        sample1,
        sample2,
        quantity2,
        rbins_absolute=rbins_absolute,
        return_counts=True,
        period=1,
    )
    result2, counts2 = radial_profile_3d(
        sample1,
        sample2,
        quantity2,
        rbins_normalized=rbins_normalized,
        normalize_rbins_by=np.zeros(npts1) + fixed_rvir,
        return_counts=True,
        period=1,
    )
    assert np.all(counts1 == counts2)
    assert np.allclose(result1, result2, rtol=0.001)


def test_radial_profile_3d_test3():
    """Create a regular mesh of ``sample1`` points and two concentric rings around
    two different points in the mesh. Give random uniform weights to the rings,
    and verify that `~halotools.mock_observables.radial_profile_3d` returns the
    correct counts and results.
    """
    npts1 = 100
    sample1 = generate_3d_regular_mesh(4)  # coords = 0.125, 0.375, 0.625, 0.875

    rbins_absolute = np.array([0.04, 0.06, 0.1])
    r1, r2 = 0.05, 0.09

    xca1, yca1, zca1 = 0.125, 0.125, 0.125
    xca2, yca2, zca2 = 0.625, 0.625, 0.625
    sample2_p1_r1 = generate_thin_shell_of_3d_points(
        npts1, r1, xca1, yca1, zca1, seed=fixed_seed, Lbox=1
    )
    sample2_p2_r1 = generate_thin_shell_of_3d_points(
        npts1, r1, xca2, yca2, zca2, seed=fixed_seed, Lbox=1
    )
    sample2_p1_r2 = generate_thin_shell_of_3d_points(
        npts1, r2, xca1, yca1, zca1, seed=fixed_seed, Lbox=1
    )
    sample2_p2_r2 = generate_thin_shell_of_3d_points(
        npts1, r2, xca2, yca2, zca2, seed=fixed_seed, Lbox=1
    )
    sample2 = np.concatenate(
        [sample2_p1_r1, sample2_p2_r1, sample2_p1_r2, sample2_p2_r2]
    )
    npts2 = len(sample2)

    with NumpyRNGContext(fixed_seed):
        inner_ring_values = np.random.uniform(-1, 1, int(npts2 / 2))
        outer_ring_values = np.random.uniform(-1, 1, int(npts2 / 2))

    quantity = np.concatenate([inner_ring_values, outer_ring_values])

    result, counts = radial_profile_3d(
        sample1,
        sample2,
        quantity,
        rbins_absolute=rbins_absolute,
        period=1,
        return_counts=True,
    )

    assert np.all(counts == npts2 / 2)
    assert np.allclose(
        result, [np.mean(inner_ring_values), np.mean(outer_ring_values)], rtol=0.001
    )


def test_radial_profile_3d_test4():
    """For two tight localizations of sample1 points each surrounded by two concentric
    shells of sample2 points, verify that both the counts and the primary result
    of `~halotools.mock_observables.radial_profile_3d` are correct. This test differs
    from test_radial_profile_3d_test2 in that here the normalize_rbins_by is operative.

    In this test, PBCs have a non-trivial impact on the results.
    """
    npts1a = 100
    xca1, yca1, zca1 = 0.3, 0.3, 0.3
    xca2, yca2, zca2 = 0.9, 0.9, 0.9

    sample1a = generate_locus_of_3d_points(npts1a, xca1, yca1, zca1, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(npts1a, xca2, yca2, zca2, seed=fixed_seed)
    sample1 = np.concatenate([sample1a, sample1b])
    npts1 = len(sample1)

    rvir = 0.013
    rvir_array = np.zeros(npts1) + rvir
    rbins_normalized = np.array([0.5, 1, 2])
    r1, r2 = 0.75 * rvir, 1.5 * rvir

    sample2_p1_r1 = generate_thin_shell_of_3d_points(
        npts1, r1, xca1, yca1, zca1, seed=fixed_seed, Lbox=1
    )
    sample2_p2_r1 = generate_thin_shell_of_3d_points(
        npts1, r1, xca2, yca2, zca2, seed=fixed_seed, Lbox=1
    )
    sample2_p1_r2 = generate_thin_shell_of_3d_points(
        npts1, r2, xca1, yca1, zca1, seed=fixed_seed, Lbox=1
    )
    sample2_p2_r2 = generate_thin_shell_of_3d_points(
        npts1, r2, xca2, yca2, zca2, seed=fixed_seed, Lbox=1
    )
    sample2 = np.concatenate(
        [sample2_p1_r1, sample2_p2_r1, sample2_p1_r2, sample2_p2_r2]
    )
    npts2 = len(sample2)

    quantity_a, quantity_b = (
        np.zeros(int(npts2 / 2)) + 0.5,
        np.zeros(int(npts2 / 2)) + 1.5,
    )
    quantity = np.concatenate([quantity_a, quantity_b])

    result, counts = radial_profile_3d(
        sample1,
        sample2,
        quantity,
        rbins_normalized=rbins_normalized,
        return_counts=True,
        period=1,
        normalize_rbins_by=rvir_array,
    )
    correct_counts = (npts1 / 2.0) * (npts2 / 2.0)
    assert np.all(counts == correct_counts)
    assert np.all(result == [0.5, 1.5])


def test_args_processing1a():
    """Verify that we correctly enforce self-consistent choices for
    all ``rbins`` arguments

    """
    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.ones(len(sample2))
    dummy_rbins = np.array([0.0001, 0.0002, 0.0003])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(sample1, sample2, quantity)
    substr = "You must either provide a ``rbins_absolute`` argument"
    assert substr in err.value.args[0]


def test_args_processing1b():
    """Verify that we correctly enforce self-consistent choices for
    all ``rbins`` arguments

    """
    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.ones(len(sample2))
    dummy_rbins = np.array([0.0001, 0.0002, 0.0003])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1,
            sample2,
            quantity,
            rbins_absolute=dummy_rbins,
            rbins_normalized=dummy_rbins,
        )
    substr = (
        "Do not provide both ``rbins_normalized`` and ``rbins_absolute`` arguments."
    )
    assert substr in err.value.args[0]


def test_args_processing1c():
    """Verify that we correctly enforce self-consistent choices for
    all ``rbins`` arguments

    """
    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.ones(len(sample2))
    dummy_rbins = np.array([0.0001, 0.0002, 0.0003])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1, sample2, quantity, rbins_absolute=dummy_rbins, normalize_rbins_by=1
        )
    substr = "you should not provide the ``normalize_rbins_by`` argument."
    assert substr in err.value.args[0]


def test_args_processing1d():
    """Verify that we correctly enforce self-consistent choices for
    all ``rbins`` arguments

    """
    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.ones(len(sample2))
    dummy_rbins = np.array([0.0001, 0.0002, 0.0003])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1, sample2, quantity, rbins_normalized=np.ones(len(sample1))
        )
    substr = "you must also provide the ``normalize_rbins_by`` argument."
    assert substr in err.value.args[0]


def test_args_processing1e():
    """Verify that we correctly enforce self-consistent choices for
    all ``rbins`` arguments

    """
    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.ones(len(sample2))
    dummy_rbins = np.array([0.000, 0.0002, 0.0003])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1,
            sample2,
            quantity,
            rbins_normalized=dummy_rbins,
            normalize_rbins_by=np.zeros(len(sample1)),
        )


def test_args_processing2():
    """Verify that we correctly enforce that ``sample1`` and ``normalize_rbins_by``
    have the same number of elements.
    """
    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.ones(len(sample2))
    rbins_normalized = np.array([0.0001, 0.0002, 0.0003])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1,
            sample2,
            quantity,
            rbins_normalized=rbins_normalized,
            normalize_rbins_by=np.ones(5),
        )
    substr = "Your input ``normalize_rbins_by`` must have the same number of elements"
    assert substr in err.value.args[0]


def test_args_processing3():
    """Verify that we correctly enforce that ``sample2`` and ``sample2_quantity``
    have the same number of elements.
    """
    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.arange(5)
    rbins_absolute = np.array([0.0001, 0.0002, 0.0003])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1, sample2, quantity, rbins_absolute=rbins_absolute
        )
    substr = "elements, but input ``sample2`` has"
    assert substr in err.value.args[0]


def test_enforce_search_length():

    npts1, npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    quantity = np.zeros(npts2)
    rbins_absolute = np.array([0.0001, 0.0002, 0.4])

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1, sample2, quantity, rbins_absolute=rbins_absolute, period=1
        )
    substr = "This exceeds the maximum permitted search length of period/3."
    assert substr in err.value.args[0]

    rbins_normalized = np.arange(1, 10)
    normalize_rbins_by = np.ones(len(sample1))

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(
            sample1,
            sample2,
            quantity,
            rbins_normalized=rbins_normalized,
            normalize_rbins_by=normalize_rbins_by,
            period=1,
        )
    substr = "This exceeds the maximum permitted search length of period/3."
    assert substr in err.value.args[0]


def test_parallel_serial_consistency():
    npts1, npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        quantity2 = np.random.random(npts2)

    rbins_absolute = np.linspace(0.01, 0.2, 5)
    fixed_rvir = 0.1
    rbins_normalized = rbins_absolute / fixed_rvir

    result1, counts1 = radial_profile_3d(
        sample1,
        sample2,
        quantity2,
        rbins_absolute=rbins_absolute,
        return_counts=True,
        period=1,
        num_threads=1,
    )
    result2, counts2 = radial_profile_3d(
        sample1,
        sample2,
        quantity2,
        rbins_absolute=rbins_absolute,
        return_counts=True,
        period=1,
        num_threads=2,
    )
    assert np.all(counts1 == counts2)
    assert np.allclose(result1, result2, rtol=0.001)


def test_pbc():
    """Regression test for Issue #862"""
    npts1, npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.uniform(0.25, 0.75, npts1 * 3).reshape((npts1, 3))
        sample2 = np.random.uniform(0.25, 0.75, npts2 * 3).reshape((npts2, 3))
        quantity2 = np.random.random(npts2)

    rbins_absolute = np.linspace(0.01, 0.2, 5)

    result1, counts1 = radial_profile_3d(
        sample1,
        sample2,
        quantity2,
        rbins_absolute=rbins_absolute,
        return_counts=True,
        period=1,
    )
    result2, counts2 = radial_profile_3d(
        sample1,
        sample2,
        quantity2,
        rbins_absolute=rbins_absolute,
        return_counts=True,
        period=None,
    )
    assert np.all(counts1 == counts2)
    assert np.allclose(result1, result2, rtol=0.001)
