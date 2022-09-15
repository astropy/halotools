""" Module providing testing for the `~halotools.mock_observables.counts_in_cylinders` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from .pure_python_counts_in_cells import (
    pure_python_counts_in_cylinders,
    pure_python_idx_in_cylinders,
)

from ..counts_in_cylinders import counts_in_cylinders

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = (
    "test_counts_in_cylinders0",
    "test_counts_in_cylinders1",
    "test_counts_in_cylinders2",
)

fixed_seed = 43
seed_list = np.arange(5).astype(int)


def test_counts_in_cylinders0():
    """For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[0,:] = (0.1, 0.1, 0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    sample2 = generate_locus_of_3d_points(
        npts2, xc=0.101, yc=0.101, zc=0.101, seed=fixed_seed
    )

    proj_search_radius, cylinder_half_length = 0.02, 0.02

    result = counts_in_cylinders(
        sample1, sample2, proj_search_radius, cylinder_half_length, period=period
    )
    assert np.shape(result) == (len(sample1),)
    assert np.sum(result) == npts2


def test_counts_in_cylinders1():
    """For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[0,:] = (0.1, 0.1, 0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    sample2 = generate_locus_of_3d_points(
        npts2, xc=0.101, yc=0.101, zc=0.101, seed=fixed_seed
    )

    proj_search_radius, cylinder_half_length = 0.02, 0.02

    result = counts_in_cylinders(
        sample1, sample2, proj_search_radius, cylinder_half_length, period=period
    )
    assert result[0] == npts2
    assert np.all(result[1:] == 0)


def test_counts_in_cylinders2():
    """For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[55,:] = (0.3,  0.5,  0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    idx_to_test = 55
    sample2 = generate_locus_of_3d_points(
        npts2,
        xc=sample1[idx_to_test, 0] + 0.001,
        yc=sample1[idx_to_test, 1] + 0.001,
        zc=sample1[idx_to_test, 2] + 0.001,
        seed=fixed_seed,
    )

    print(
        "Sample2 (xmin, xmax) = ({0:.3F}, {1:.3F})".format(
            np.min(sample2[:, 0]), np.max(sample2[:, 0])
        )
    )
    print(
        "Sample2 (ymin, ymax) = ({0:.3F}, {1:.3F})".format(
            np.min(sample2[:, 1]), np.max(sample2[:, 1])
        )
    )
    print(
        "Sample2 (zmin, zmax) = ({0:.3F}, {1:.3F})".format(
            np.min(sample2[:, 2]), np.max(sample2[:, 2])
        )
    )

    proj_search_radius, cylinder_half_length = 0.02, 0.02

    result = counts_in_cylinders(
        sample1, sample2, proj_search_radius, cylinder_half_length, period=period
    )
    assert np.sum(result == npts2)

    idx_result = np.where(result != 0)[0]
    print(
        "Index where the counts_in_cylinders function identifies points = {0}".format(
            idx_result
        )
    )
    print("Correct index should be {0}".format(idx_to_test))
    print("\n")
    print(
        "Point in sample1 corresponding to the incorrect index = {0}".format(
            sample1[idx_result[0]]
        )
    )
    print(
        "Point in sample1 corresponding to the correct index   = {0}".format(
            sample1[idx_to_test]
        )
    )
    assert result[idx_to_test] == npts2
    assert np.all(result[0:idx_to_test] == 0)
    assert np.all(result[idx_to_test + 1 :] == 0)


def test_counts_in_cylinders_brute_force1():
    """"""
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))

        rp_max = np.zeros(npts1) + 0.2
        pi_max = np.zeros(npts1) + 0.2
        brute_force_result = pure_python_counts_in_cylinders(
            sample1, sample2, rp_max, pi_max
        )
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_brute_force2():
    """"""
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))

        rp_max = np.zeros(npts1) + 0.2
        pi_max = np.zeros(npts1) + 0.2
        brute_force_result = pure_python_counts_in_cylinders(
            sample1, sample2, rp_max, pi_max, period=1
        )
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_brute_force3():
    """"""
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))
            rp_max = np.random.uniform(0, 0.2, npts1)
            pi_max = np.random.uniform(0, 0.2, npts1)

        brute_force_result = pure_python_counts_in_cylinders(
            sample1, sample2, rp_max, pi_max
        )
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_brute_force4():
    """"""
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))
            rp_max = np.random.uniform(0, 0.2, npts1)
            pi_max = np.random.uniform(0, 0.2, npts1)

        brute_force_result = pure_python_counts_in_cylinders(
            sample1, sample2, rp_max, pi_max, period=1
        )
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_error_handling():
    """"""
    npts1 = 100
    npts2 = 90

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        rp_max = np.random.uniform(0, 0.2, npts1)
        pi_max = np.random.uniform(0, 0.2, npts1)

    with pytest.raises(ValueError) as err:
        __ = counts_in_cylinders(sample1, sample2, rp_max[1:], pi_max, period=1)
    substr = "Input ``proj_search_radius`` must be a scalar or length-Npts1 array"
    assert substr in err.value.args[0]

    with pytest.raises(ValueError) as err:
        __ = counts_in_cylinders(sample1, sample2, rp_max, pi_max[1:], period=1)
    substr = "Input ``cylinder_half_length`` must be a scalar or length-Npts1 array"
    assert substr in err.value.args[0]

    __ = counts_in_cylinders(
        sample1,
        sample2,
        rp_max,
        pi_max,
        period=1,
        approx_cell1_size=0.2,
        approx_cell2_size=0.2,
    )


def test_counts_in_cylinders_pbc():
    npts1 = 1000
    npts2 = 9000

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.uniform(0.25, 0.75, npts1 * 3).reshape((npts1, 3))
            sample2 = np.random.uniform(0.25, 0.75, npts2 * 3).reshape((npts2, 3))
            rp_max = np.random.uniform(0, 0.1, npts1)
            pi_max = np.random.uniform(0, 0.1, npts1)

        result_pbc = counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1)
        result_nopbc = counts_in_cylinders(
            sample1, sample2, rp_max, pi_max, period=None
        )
        assert np.allclose(result_pbc, result_nopbc)


def test_counts_in_cylinders_parallel_serial_consistency():
    """Enforce that the counts-in-cylinder function returns identical results
    when called in serial or parallel.

    This is a regression test for Issue #908,
    https://github.com/astropy/halotools/issues/908.
    """
    period = 1
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    sample2 = generate_locus_of_3d_points(npts2, xc=0.101, yc=0.101, zc=0.101)

    proj_search_radius, cylinder_half_length = 0.02, 0.02

    result1 = counts_in_cylinders(
        sample1, sample2, proj_search_radius, cylinder_half_length, period=period
    )
    result2 = counts_in_cylinders(
        sample1,
        sample2,
        proj_search_radius,
        cylinder_half_length,
        period=period,
        num_threads=2,
    )

    assert result1.shape == result2.shape


@pytest.mark.parametrize("num_threads", [1, 2])
def test_counts_in_cylinders_with_indexes(num_threads):
    """"""
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))

        rp_max = np.zeros(npts1) + 0.2
        pi_max = np.zeros(npts1) + 0.2
        brute_force_indexes = pure_python_idx_in_cylinders(
            sample1, sample2, rp_max, pi_max
        )
        brute_force_counts = pure_python_counts_in_cylinders(
            sample1, sample2, rp_max, pi_max
        )
        counts, indexes = counts_in_cylinders(
            sample1,
            sample2,
            rp_max,
            pi_max,
            return_indexes=True,
            num_threads=num_threads,
        )

        assert np.all(_sort(indexes) == _sort(brute_force_indexes))
        assert np.all(counts == brute_force_counts)
        assert len(indexes) > npts1  # assert that we have tested array resizing


def test_counts_in_cylinders_single_index():
    sample1 = np.random.random((1, 3))
    sample2 = sample1
    rp_max = np.zeros(1) + 0.2
    pi_max = np.zeros(1) + 0.2
    counts, indexes = counts_in_cylinders(
        sample1, sample2, rp_max, pi_max, return_indexes=True
    )
    assert len(indexes) == 1 and counts == np.array([1])


def test_counts_in_cylinders_indexes_no_match():
    sample1 = np.random.random((1, 3))
    sample2 = sample1 + 0.2
    rp_max = np.zeros(1) + 0.02
    pi_max = np.zeros(1) + 0.02
    counts, indexes = counts_in_cylinders(
        sample1, sample2, rp_max, pi_max, return_indexes=True
    )
    assert len(indexes) == 0 and counts == np.array([0])


def test_counts_in_cylinders_autocorr0():
    period, rp_max, pi_max = 1, 0.1, 0.1
    npts = 100
    sample = generate_locus_of_3d_points(
        npts, xc=0.101, yc=0.101, zc=0.101, seed=fixed_seed
    )
    counts = counts_in_cylinders(sample, None, rp_max, pi_max, period)
    assert np.all(counts == npts - 1)

    counts, indexes = counts_in_cylinders(
        sample, None, rp_max, pi_max, period, return_indexes=True
    )
    assert np.all(counts == npts - 1)
    assert len(indexes) == (npts - 1) * npts


def test_counts_in_cylinders_autocorr1():
    npts = 100

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample = np.random.random((npts, 3))

        rp_max = np.zeros(npts) + 0.2
        pi_max = np.zeros(npts) + 0.2
        brute_force_counts = pure_python_counts_in_cylinders(
            sample, None, rp_max, pi_max
        )
        brute_force_indexes = pure_python_idx_in_cylinders(sample, None, rp_max, pi_max)
        counts, indexes = counts_in_cylinders(
            sample, None, rp_max, pi_max, return_indexes=True
        )
        assert counts.shape == brute_force_counts.shape
        assert np.all(counts == brute_force_counts)
        assert np.all(_sort(indexes) == _sort(brute_force_indexes))


def _sort(indexes):
    return np.sort(indexes, order=["i1", "i2"])


def test_counts_in_cylinders_condition_true():
    npair, indexes = _mass_frac_tester(True)
    assert np.all(npair == 1)
    assert np.all(indexes["i1"] == indexes["i2"])


def test_counts_in_cylinders_condition_false():
    npair, indexes = _mass_frac_tester(False)
    assert np.all(npair == 0)
    assert len(indexes) == 0


def _mass_frac_tester(equality):
    sample1 = sample2 = np.arange(30).reshape(10, 3)
    proj_search_radius = 1.0
    cylinder_half_length = 1.0

    m1 = np.logspace(-20, 20, 10)
    m2 = m1 / 2.0
    lim = (0.5, 0.5)
    lower_equality = upper_equality = equality
    args = (m1, m2, lim, lower_equality, upper_equality)

    return counts_in_cylinders(
        sample1,
        sample2,
        proj_search_radius,
        cylinder_half_length,
        return_indexes=True,
        condition="mass_frac",
        condition_args=args,
    )
