#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 

from ..spherical_isolation import spherical_isolation

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ['test_spherical_isolation_criteria1']

sample1 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1)

def test_spherical_isolation_criteria1():
    sample2 = generate_locus_of_3d_points(100, xc=0.5)
    r_max = 0.1
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)

def test_spherical_isolation_criteria2():
    sample2a = generate_locus_of_3d_points(100, xc=0.11)
    sample2b = generate_locus_of_3d_points(100, xc=1.11)
    sample2 = np.concatenate((sample2a, sample2b))
    r_max = 0.3
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == False)

def test_spherical_isolation_criteria3():
    sample2 = generate_locus_of_3d_points(100, xc=0.95)
    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == False)
    iso2 = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso2 == True)

def test_spherical_isolation_grid1():
    """ Create a regular grid inside the unit box with points on each of the following 
    nodes: 0.1, 0.3, 0.5, 0.7, 0.9. Demonstrate that all points in such a sample 
    are isolated if r_max < 0.2, regardless of periodic boundary conditions. 
    """
    sample1 = generate_3d_regular_mesh(5) 

    r_max = 0.1
    iso = spherical_isolation(sample1, sample1, r_max)
    assert np.all(iso == True)
    iso = spherical_isolation(sample1, sample1, r_max, period=1)
    assert np.all(iso == True)

    r_max = 0.25
    iso2 = spherical_isolation(sample1, sample1, r_max)
    assert np.all(iso2 == False)
    iso2 = spherical_isolation(sample1, sample1, r_max, period=1)
    assert np.all(iso2 == False)


def test_spherical_isolation_grid2():
    sample1 = generate_3d_regular_mesh(5)
    sample2 = generate_3d_regular_mesh(10)

    r_max = 0.001
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == False)

    r_max = 0.001
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == False)

def test_shifted_randoms():
    npts = 1e3
    sample1 = np.random.random((npts, 3))
    epsilon = 0.001
    sample2 = sample1 + epsilon

    r_max = epsilon/10.
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)

    r_max = 2*epsilon
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == False)

def test_shifted_mesh():
    npts = 1e3
    sample1 = generate_3d_regular_mesh(10)
    epsilon = 0.001
    sample2 = sample1 + epsilon

    r_max = epsilon/10.
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)

    r_max = 2*epsilon
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == False)
