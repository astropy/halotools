""" Module providing testing for the `~halotools.mock_observables.spherical_isolation` function.
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np

from ..cylindrical_isolation import cylindrical_isolation
from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ('test_cylindrical_isolation1', 'test_cylindrical_isolation2',
    'test_cylindrical_isolation3', 'test_cylindrical_isolation4',
    'test_cylindrical_isolation5', 'test_cylindrical_isolation_indices')

fixed_seed = 43


def test_cylindrical_isolation1():
    """ Verify that the `~halotools.mock_observables.cylindrical_isolation` function
    returns all points as isolated for two distant localizations of points.
    """
    sample1 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(100, xc=0.9, seed=fixed_seed)
    pi_max = 0.1
    rp_max = 0.1
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
    assert np.all(iso == True)

    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
    assert np.all(iso == True)


def test_cylindrical_isolation2():
    """ Verify that the `~halotools.mock_observables.cylindrical_isolation` function
    returns no points as isolated when a subset of ``sample2`` lies within ``sample1``
    """
    sample1 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    pi_max = 0.1
    rp_max = 0.1
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
    assert np.all(iso == False)


def test_cylindrical_isolation3():
    """ Verify that the `~halotools.mock_observables.cylindrical_isolation` function
    returns the correct subset of points in ``sample1`` as being isolated.

    For this test, PBCs are irrelevant.
    """
    sample1a = generate_locus_of_3d_points(100, xc=0.11, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(100, xc=1.11, seed=fixed_seed)
    sample1 = np.concatenate((sample1a, sample1b))

    sample2 = generate_locus_of_3d_points(100, xc=0.11, yc=0.1, zc=0.1, seed=fixed_seed)
    pi_max = 0.1
    rp_max = 0.1
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)

    assert np.all(iso[:100] == False)
    assert np.all(iso[100:] == True)


def test_cylindrical_isolation4():
    """ Verify that the `~halotools.mock_observables.cylindrical_isolation` function
    returns the correct subset of points in ``sample1`` as being isolated.

    For this test, PBCs have a non-trivial impact on the result.
    """
    sample1a = generate_locus_of_3d_points(100, xc=0.5, yc=0.1, zc=0.1, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(100, xc=0.99, yc=0.1, zc=0.1, seed=fixed_seed)
    sample1 = np.concatenate((sample1a, sample1b))

    sample2 = generate_locus_of_3d_points(100, xc=0.11, yc=0.1, zc=0.1, seed=fixed_seed)

    pi_max = 0.2
    rp_max = 0.2
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
    assert np.all(iso == True)

    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
    assert np.all(iso[:100] == True)
    assert np.all(iso[100:] == False)


def test_cylindrical_isolation5():
    """ For two tight localizations of distant points,
    verify that the `~halotools.mock_observables.cylindrical_isolation` function
    has independently correct behavior in all combinations of limits for pi_max and rp_max
    """
    sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95, seed=fixed_seed)

    rp_max, pi_max = 0.2, 0.2
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
    assert np.all(iso == True)
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1])
    assert np.all(iso == False)

    rp_max, pi_max = 0.2, 0.2
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1000, 1000, 1])
    assert np.all(iso == True)
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1000])
    assert np.all(iso == True)

    rp_max, pi_max = 0.05, 0.2
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
    assert np.all(iso == True)
    rp_max, pi_max = 0.2, 0.05
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
    assert np.all(iso == True)


def test_cylindrical_isolation_indices():
    """ Create two regular meshes such that all points in the meshes are isolated from each other.
    Insert a single point into mesh1 that is immediately adjacent to one of the points in mesh2.
    Verify that there is only a single isolated point and that it has the correct index.
    """

    sample1_mesh = generate_3d_regular_mesh(5)  # 0.1, 0.3, 0.5, 0.7, 0.9
    sample2 = generate_3d_regular_mesh(10)  # 0.05, 0.15, 0.25, 0.35, ..., 0.95

    insertion_idx = 5
    sample1 = np.insert(sample1_mesh, insertion_idx*3, [0.06, 0.06, 0.06]).reshape((len(sample1_mesh)+1, 3))

    rp_max, pi_max = 0.025, 0.025
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
    correct_result = np.ones(len(iso))
    correct_result[insertion_idx] = 0
    assert np.all(iso == correct_result)
