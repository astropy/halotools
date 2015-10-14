#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
#load comparison simple pair counters
from ..pairs import npairs as simp_npairs
from ..pairs import wnpairs as simp_wnpairs
#load rect_cuboid_pairs pair counters
from ..double_tree_pairs import double_tree_npairs

np.random.seed(1)

def test_double_tree_npairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T

    rbins = np.array([0.0,0.1,0.2,0.3])

    result = double_tree_npairs(data1, data1, rbins, period)
    
    test_result = simp_npairs(data1, data1, rbins, period=period)

    assert np.all(test_result==result), "pair counts for PBC-case are incorrect"

def test_double_tree_npairs_nonperiodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.array([0.0,0.1,0.2,0.3])

    result = double_tree_npairs(data1, data1, rbins, period=None)
    test_result = simp_npairs(data1, data1, rbins, period=None)
    
    assert np.all(test_result==result), "pair counts for non-PBC case are incorrect"
