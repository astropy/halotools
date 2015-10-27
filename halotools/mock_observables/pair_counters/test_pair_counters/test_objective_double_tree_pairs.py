#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
#load comparison simple pair counters
from ..pairs import wnpairs as simp_wnpairs
#load rect_cuboid_pairs pair counters
from ..objective_double_tree_pairs import obj_wnpairs as wnpairs


def test_wnpairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3])

    result = wnpairs(data1, data1, rbins, period=period, weights1=weights1, weights2=weights1, wfunc=1)
    
    test_result = simp_wnpairs(data1, data1, rbins, period=period, weights1=weights1, weights2=weights1)

    print(test_result,result)
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


def test_wnpairs_nonperiodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3])

    result = wnpairs(data1, data1, rbins, period=None, weights1=weights1, weights2=weights1, wfunc=1)
    
    test_result = simp_wnpairs(data1, data1, rbins, period=None, weights1=weights1, weights2=weights1)

    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"

