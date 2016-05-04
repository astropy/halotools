#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
    unicode_literals)
    
import numpy as np
import pytest 
import numpy as np
from scipy.sparse import coo_matrix

from ..pairwise_distance_3d import pairwise_distance_3d

from ...tests.cf_helpers import generate_locus_of_3d_points
from ...tests.cf_helpers import generate_3d_regular_mesh

__all__ = ["test_pairwise_distance_3d_periodic_mesh_grid_1",
           "test_pairwise_distance_3d_nonperiodic_mesh_grid_1",
           "test_pairwise_distance_3d_nonperiodic_random_1",
           "test_pairwise_distance_3d_periodic_tight_locus1",
           "test_pairwise_distance_3d_nonperiodic_tight_locus1",
           "test_pairwise_distance_3d_nonperiodic_tight_locus2"
           ]

def test_pairwise_distance_3d_periodic_mesh_grid_1():
    """
    test that each point is paired with its neighbor on a regular grid
    """
    #regular grid
    Npts_per_dim = 10
    mesh_sample = generate_3d_regular_mesh(Npts_per_dim)
    period = 1.0
    
    #test on a uniform grid
    rmax=0.10001
    m = pairwise_distance_3d(mesh_sample, mesh_sample, rmax, period=period)
    
    #each point has 7 connections including 1 self connection
    #N = (10^3)*7
    assert m.getnnz()==7000
    
    #diagonal self matches should have distance 0
    assert np.all(m.diagonal()==0.0)
    
    #off diagonal should all be 0.1
    i,j = m.nonzero()
    assert np.allclose(np.asarray(m.todense()[i,j]),0.1)


def test_pairwise_distance_3d_nonperiodic_mesh_grid_1():
    """
    test that each point is paired with its neighbor on a regular grid
    """
    #regular grid
    Npts_per_dim = 10
    mesh_sample = generate_3d_regular_mesh(Npts_per_dim)
    period = 1.0
    
    #test on a uniform grid
    rmax=0.10001
    m = pairwise_distance_3d(mesh_sample, mesh_sample, rmax, period=None)
    
    #each point has 7 connections including 1 self connection
    #N = (10^3)*7
    #points on the 6 faces have fewer connections N - (10*10*6) = 6400
    #overlapping accounts for edges (2 less) and corners (3 less)
    assert m.getnnz()==6400
    
    #diagonal self matches should have distance 0
    assert np.all(m.diagonal()==0.0)
    
    #off diagonal should all be 0.1
    i,j = m.nonzero()
    assert np.allclose(np.asarray(m.todense()[i,j]),0.1)


def test_pairwise_distance_3d_nonperiodic_random_1():
    """
    test that each point is paired with every point when rmax is large enough
    """
    #random points
    Npts = 10
    random_sample = np.random.random((Npts,3))
    period = 1.0
    
    rmax=10.0
    m = pairwise_distance_3d(random_sample, random_sample, rmax, period=None)
    
    #each point is paired with every other point
    assert m.getnnz()==Npts**2


def test_pairwise_distance_3d_periodic_tight_locus1():
    """
    test that each point in two loci of points are paired across PBCs when rmax 
    is large enough
    """
    #tigh locus
    Npts1 = 10
    Npts2 = 10
    data1 = generate_locus_of_3d_points(Npts1, xc=0.05, yc=0.05, zc=0.05)
    data2 = generate_locus_of_3d_points(Npts2, xc=0.95, yc=0.95, zc=0.95)
    period = 1.0
    
    #should be no connections
    rmax=0.01
    m = pairwise_distance_3d(data1, data2, rmax, period=period)
    
    #each point has 0 connections including 1 self connection
    assert m.getnnz()==0
    
    #should be 10*10 connections
    rmax=0.3
    m = pairwise_distance_3d(data1, data2, rmax, period=period)
    
    #each point has 10 connections
    assert m.getnnz()== Npts1*Npts2


def test_pairwise_distance_3d_nonperiodic_tight_locus1():
    """
    test that each point in two loci of points are NOT paired across PBCs
    """
    #tight locus
    Npts1 = 10
    Npts2 = 10
    data1 = generate_locus_of_3d_points(Npts1, xc=0.05, yc=0.05, zc=0.05)
    data2 = generate_locus_of_3d_points(Npts2, xc=0.95, yc=0.95, zc=0.95)
    period = 1.0
    
    #should be no connections
    rmax=0.01
    m = pairwise_distance_3d(data1, data2, rmax, period=None)
    
    #each point has 0 connections including 1 self connection
    assert m.getnnz()==0
    
    #should be 0 connections
    rmax=0.3
    m = pairwise_distance_3d(data1, data2, rmax, period=None)
    
    #each point has 0 connections
    assert m.getnnz()== 0


def test_pairwise_distance_3d_nonperiodic_tight_locus2():
    """
    test that pairs of points in two loci are paired when rmax is large enough
    """
    #tight locus
    Npts1 = 10
    Npts2 = 10
    data1 = generate_locus_of_3d_points(Npts1, xc=0.5, yc=0.5, zc=0.5)
    data2 = generate_locus_of_3d_points(Npts2, xc=0.6, yc=0.6, zc=0.6)
    period = 1.0
    
    #should be no connections
    rmax=0.01
    m = pairwise_distance_3d(data1, data2, rmax, period=None)
    
    #each point has 0 connections including 1 self connection
    assert m.getnnz()==0
    
    #should 10
    rmax=0.2
    m = pairwise_distance_3d(data1, data2, rmax, period=None)
    
    #each point has 0 connections including 1 self connection
    assert m.getnnz()==Npts1*Npts2
