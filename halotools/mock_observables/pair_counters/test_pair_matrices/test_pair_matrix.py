from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
slow = pytest.mark.slow

import numpy as np
import scipy
from scipy import spatial
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

#load comparison simple pair counters
from ..double_tree_pair_matrix import pair_matrix, xy_z_pair_matrix

__all__ = ['test_pair_matrix_periodic','test_pair_matrix_non_periodic',\
        'test_xy_z_pair_matrix_periodic','test_xy_z_pair_matrix_non_periodic']

#create some toy data to test functions
Npts = 1e4
Lbox = np.array([1.0,1.0,1.0])
period = Lbox
#random data
x = np.random.uniform(0, Lbox[0], Npts)
y = np.random.uniform(0, Lbox[1], Npts)
z = np.random.uniform(0, Lbox[2], Npts)
data1 = np.vstack((x,y,z)).T
#uniform grid
x = np.arange(0.0,1.0,0.1) #don't change
x,y,z = np.meshgrid(x, x, x)
x = x.flatten()
y = y.flatten()
z = z.flatten()
data2 = np.vstack((x,y,z)).T

@slow
def test_pair_matrix_periodic():
    
    r_max = 0.1
    m = pair_matrix(data1, data1, r_max, period=period)
    assert isinstance(m,scipy.sparse.coo.coo_matrix)
    
    #test on a uniform grid
    r_max=0.10001
    m = pair_matrix(data2, data2, r_max, period=period)

    #each point has 7 connections including 1 self connection
    #includes self connections
    # N = (10^3)*7
    assert m.getnnz()==7000

@slow
def test_pair_matrix_non_periodic():
    
    r_max = 0.1
    m = pair_matrix(data1, data1, r_max, period=None)
    assert isinstance(m,scipy.sparse.coo.coo_matrix)
    
    #test on a uniform grid
    r_max=0.10001
    m = pair_matrix(data2, data2, r_max, period=None)

    # connections: inside + faces + edges + corners
    # includes self connections
    # N = (8^3)*7 + (8^2*6)*6 + (12*8)*5 + (8)*4 
    assert m.getnnz()==6400
    
    #calculate all distances and compare to scipy.distance
    #r_max=2.0
    #m = pair_matrix(data1, data1, r_max, period=None, Lbox=Lbox)
    #m = m.todense()
    
    #mm = spatial.distance.cdist(data1,data1)
    #mm = coo_matrix(mm)
    #mm = mm.todense()
    
    #assert np.all(m==mm)

@slow
def test_xy_z_pair_matrix_periodic():
    
    rp_max=0.01
    pi_max=0.01
    approx_cell1_size = [0.1,0.1,0.1]
    approx_cell2_size = approx_cell1_size
    
    m_perp, m_para = xy_z_pair_matrix(data1, data1, rp_max, pi_max, period=period,
                                      approx_cell1_size = approx_cell1_size,
                                      approx_cell2_size = approx_cell2_size)
    assert isinstance(m_perp,scipy.sparse.coo.coo_matrix)
    assert isinstance(m_para,scipy.sparse.coo.coo_matrix)
    
    #test on a uniform grid
    rp_max=0.11
    pi_max=0.11
    
    m_perp, m_para = xy_z_pair_matrix(data2, data2, rp_max, pi_max, period=period)

    #each point has 7 connections including 1 self connection
    #includes self connections
    #N = (10^3)*15
    assert m_para.getnnz()==15000
    assert m_perp.getnnz()==15000

@slow
def test_xy_z_pair_matrix_non_periodic():
    
    rp_max=0.01
    pi_max=0.01
    approx_cell1_size = [0.1,0.1,0.1]
    approx_cell2_size = approx_cell1_size
    
    m_perp, m_para = xy_z_pair_matrix(data1, data1, rp_max, pi_max, period=None,
                                      approx_cell1_size = approx_cell1_size,
                                      approx_cell2_size = approx_cell2_size)
    
    assert isinstance(m_perp,scipy.sparse.coo.coo_matrix)
    assert isinstance(m_para,scipy.sparse.coo.coo_matrix)
    
    #test on a uniform grid
    rp_max=0.1001
    pi_max=0.1001
    
    m_perp, m_para = xy_z_pair_matrix(data2, data2, rp_max, pi_max, period=None)

    # connections: inside + faces + edges + corners
    # includes self connections
    assert m_perp.getnnz()==12880
    assert m_para.getnnz()==12880
    
    
