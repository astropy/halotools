#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 

from ..cylindrical_isolation import cylindrical_isolation
from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ['test_cylindrical_isolation1']

sample1 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1)


def test_cylindrical_isolation1():
    """ For two tight localizations of points right on top of each other, 
    all points in sample1 should not be isolated. 
    """
    sample2 = generate_locus_of_3d_points(100)
    pi_max = 0.1
    rp_max = 0.1
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
    assert np.all(iso == False)

def test_cylindrical_isolation2():
    """ For two tight localizations of distant points, 
    all points in sample1 should be isolated unless PBCs are turned on
    """
    sample1 = generate_locus_of_3d_points(100, xc=0.05, yc=0.05, zc=0.05)
    sample2 = generate_locus_of_3d_points(100, xc=0.95, yc=0.95, zc=0.95)
    pi_max = 0.2
    rp_max = 0.2
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
    assert np.all(iso == True)
    iso2 = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1.)
    assert np.all(iso2 == False)

def test_cylindrical_isolation3():
    """ For two tight localizations of distant points, 
    verify independently correct behavior for pi_max and rp_max
    """
    sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
    sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)

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

    sample1_mesh = generate_3d_regular_mesh(5) # 0.1, 0.3, 0.5, 0.7, 0.9
    sample2 = generate_3d_regular_mesh(10) # 0.05, 0.15, 0.25, 0.35, ..., 0.95

    insertion_idx = 5
    sample1 = np.insert(sample1_mesh, insertion_idx*3, [0.06, 0.06, 0.06]).reshape((len(sample1_mesh)+1, 3))

    rp_max, pi_max = 0.025, 0.025
    iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
    correct_result = np.ones(len(iso))
    correct_result[insertion_idx] = 0
    assert np.all(iso == correct_result)

