#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from ..pairs import wnpairs as pure_python_weighted_pairs
from ..marked_double_tree_pairs import marked_npairs
from ..double_tree_pairs import npairs


def test_marked_npairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    #weights1 = np.ones(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3])

    result = marked_npairs(data1, data1, rbins, period=period, weights1=weights1, weights2=weights1, wfunc=1)
    
    test_result = pure_python_weighted_pairs(data1, data1, rbins, period=period, weights1=weights1, weights2=weights1)

    print(test_result,result)
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


def test_marked_npairs_nonperiodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    #weights1 = np.ones(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3])

    result = marked_npairs(data1, data1, rbins, period=None, weights1=weights1, weights2=weights1, wfunc=1)
    
    test_result = pure_python_weighted_pairs(data1, data1, rbins, period=None, weights1=weights1, weights2=weights1)
    
    print(test_result,result)
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


def test_marked_npairs_wfuncs():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)-0.5
    #weights1 = np.ones(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3])

    result = marked_npairs(data1, data1, rbins, period=period, weights1=weights1, weights2=weights1, wfunc=1)
    test_result = pure_python_weighted_pairs(data1, data1, rbins, period=period, weights1=weights1, weights2=weights1)
    
    print(result)
    print(test_result)
    print(result/test_result)
    
    assert np.all(np.isclose(result,test_result,rtol=0.0001)), "wfunc 1 returned incorrect results"

