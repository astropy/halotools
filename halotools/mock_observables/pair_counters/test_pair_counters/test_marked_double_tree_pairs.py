#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest 
from copy import copy 

from ..pairs import wnpairs as pure_python_weighted_pairs
from ..marked_double_tree_pairs import marked_npairs
from ..marked_double_tree_helpers import _func_signature_int_from_wfunc
from ..double_tree_pairs import npairs

from ....custom_exceptions import HalotoolsError

def test_marked_npairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3])

    result = marked_npairs(data1, data1, 
        rbins, period=period, weights1=weights1, weights2=weights1, wfunc=1)
    
    test_result = pure_python_weighted_pairs(data1, data1, rbins, 
        period=period, weights1=weights1, weights2=weights1)

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

    result = marked_npairs(data1, data1, rbins, period=None, 
        weights1=weights1, weights2=weights1, wfunc=1)
    
    test_result = pure_python_weighted_pairs(data1, data1, rbins, 
        period=None, weights1=weights1, weights2=weights1)
    
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


def test_marked_npairs_wfuncs_signatures():
    """ Loop over all wfuncs and ensure that the wfunc signature is handled correctly. 
    """
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)

    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.array([0.0,0.1,0.2,0.3])


    # Determine how many wfuncs have currently been implemented
    wfunc_index = 1
    while True:
        try:
            _ = _func_signature_int_from_wfunc(wfunc_index)
            wfunc_index += 1
        except HalotoolsError:
            break
    num_wfuncs = copy(wfunc_index)

    # Now loop over all all available wfunc indices
    for wfunc_index in xrange(1, num_wfuncs):
        signature = _func_signature_int_from_wfunc(wfunc_index)
        weights = np.random.random(Npts*signature).reshape(Npts, signature) - 0.5
        result = marked_npairs(data1, data1, rbins, period=period, 
            weights1=weights, weights2=weights, wfunc=wfunc_index)

        with pytest.raises(HalotoolsError):
            signature = _func_signature_int_from_wfunc(wfunc_index) + 1
            weights = np.random.random(Npts*signature).reshape(Npts, signature) - 0.5
            result = marked_npairs(data1, data1, rbins, period=period, 
                weights1=weights, weights2=weights, wfunc=wfunc_index)


def test_marked_npairs_wfuncs_behavior():
    """ Verify the behavior of a few wfunc-weighted counters by comparing pure python, unmarked 
    pairs to the returned result from a uniformly weighted set of points.  
    """

    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)

    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T

    rbins = np.array([0.0,0.1,0.2,0.3])

    test_result = pure_python_weighted_pairs(data1, data1, 
        rbins, period=period)

    # wfunc = 1
    weights = np.ones(Npts)*3
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=1)
    assert np.all(result == 9.*test_result)

    # wfunc = 2
    weights = np.ones(Npts)*3
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=2)
    assert np.all(result == 6.*test_result)

    # wfunc = 3
    weights2 = np.ones(Npts)*2
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=3)
    assert np.all(result == 9.*test_result)

    weights = np.vstack([weights3, weights2]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=3)
    assert np.all(result == 4.*test_result)

    # wfunc = 4
    weights2 = np.ones(Npts)*2
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=4)
    assert np.all(result == 0)

    # wfunc = 5
    weights2 = np.ones(Npts)*2
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=5)
    assert np.all(result == 0)

    # wfunc = 6
    weights2 = np.ones(Npts)
    weights3 = np.zeros(Npts)-1

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=6)
    assert np.all(result == -test_result)

    # wfunc = 7
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=7)
    assert np.all(result == 3*test_result)

    # wfunc = 8
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=8)
    assert np.all(result == 3*test_result)

    # wfunc = 9
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=9)
    assert np.all(result == 0)

    weights2 = np.ones(Npts)
    weights3 = -np.ones(Npts)*3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs(data1, data1, rbins, period=period, 
    weights1=weights, weights2=weights, wfunc=9)
    assert np.all(result == -3*test_result)







