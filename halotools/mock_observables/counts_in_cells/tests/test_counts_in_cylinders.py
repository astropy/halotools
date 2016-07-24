""" Module providing testing for the `~halotools.mock_observables.counts_in_cylinders` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from astropy.tests.helper import pytest

from .pure_python_counts_in_cells import pure_python_counts_in_cylinders

from ..counts_in_cylinders import counts_in_cylinders

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ('test_counts_in_cylinders0', 'test_counts_in_cylinders1', 'test_counts_in_cylinders2')

fixed_seed = 43
seed_list = np.arange(5).astype(int)


def test_counts_in_cylinders0():
    """ For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[0,:] = (0.1, 0.1, 0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    sample2 = generate_locus_of_3d_points(npts2, xc=0.101, yc=0.101, zc=0.101, seed=fixed_seed)

    proj_search_radius, cylinder_half_length = 0.02, 0.02

    result = counts_in_cylinders(sample1, sample2, proj_search_radius, cylinder_half_length, period=period)
    assert np.shape(result) == (len(sample1), )
    assert np.sum(result) == npts2


def test_counts_in_cylinders1():
    """ For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[0,:] = (0.1, 0.1, 0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    sample2 = generate_locus_of_3d_points(npts2, xc=0.101, yc=0.101, zc=0.101, seed=fixed_seed)

    proj_search_radius, cylinder_half_length = 0.02, 0.02

    result = counts_in_cylinders(sample1, sample2, proj_search_radius, cylinder_half_length, period=period)
    assert result[0] == npts2
    assert np.all(result[1:] == 0)


def test_counts_in_cylinders2():
    """ For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[55,:] = (0.3,  0.5,  0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    idx_to_test = 55
    sample2 = generate_locus_of_3d_points(npts2,
        xc=sample1[idx_to_test, 0] + 0.001,
        yc=sample1[idx_to_test, 1] + 0.001,
        zc=sample1[idx_to_test, 2] + 0.001,
        seed=fixed_seed)

    print("Sample2 (xmin, xmax) = ({0:.3F}, {1:.3F})".format(np.min(sample2[:, 0]), np.max(sample2[:, 0])))
    print("Sample2 (ymin, ymax) = ({0:.3F}, {1:.3F})".format(np.min(sample2[:, 1]), np.max(sample2[:, 1])))
    print("Sample2 (zmin, zmax) = ({0:.3F}, {1:.3F})".format(np.min(sample2[:, 2]), np.max(sample2[:, 2])))

    proj_search_radius, cylinder_half_length = 0.02, 0.02

    result = counts_in_cylinders(sample1, sample2, proj_search_radius, cylinder_half_length, period=period)
    assert np.sum(result == npts2)

    idx_result = np.where(result != 0)[0]
    print("Index where the counts_in_cylinders function identifies points = {0}".format(idx_result))
    print("Correct index should be {0}".format(idx_to_test))
    print("\n")
    print("Point in sample1 corresponding to the incorrect index = {0}".format(sample1[idx_result[0]]))
    print("Point in sample1 corresponding to the correct index   = {0}".format(sample1[idx_to_test]))
    assert result[idx_to_test] == npts2
    assert np.all(result[0:idx_to_test] == 0)
    assert np.all(result[idx_to_test+1:] == 0)


def test_counts_in_cylinders_brute_force1():
    """
    """
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))

        rp_max = np.zeros(npts1) + 0.2
        pi_max = np.zeros(npts1) + 0.2
        brute_force_result = pure_python_counts_in_cylinders(sample1, sample2, rp_max, pi_max)
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_brute_force2():
    """
    """
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))

        rp_max = np.zeros(npts1) + 0.2
        pi_max = np.zeros(npts1) + 0.2
        brute_force_result = pure_python_counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1)
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_brute_force3():
    """
    """
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))
            rp_max = np.random.uniform(0, 0.2, npts1)
            pi_max = np.random.uniform(0, 0.2, npts1)

        brute_force_result = pure_python_counts_in_cylinders(sample1, sample2, rp_max, pi_max)
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_brute_force4():
    """
    """
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))
            rp_max = np.random.uniform(0, 0.2, npts1)
            pi_max = np.random.uniform(0, 0.2, npts1)

        brute_force_result = pure_python_counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1)
        result = counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1)
        assert brute_force_result.shape == result.shape
        assert np.all(result == brute_force_result)


def test_counts_in_cylinders_error_handling():
    """
    """
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

    __ = counts_in_cylinders(sample1, sample2, rp_max, pi_max, period=1,
        approx_cell1_size=0.2, approx_cell2_size=0.2)
