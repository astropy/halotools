#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest 

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

    for wfunc_index in xrange(1, 12):
        signature = _func_signature_int_from_wfunc(wfunc_index)
        weights = np.random.random(Npts*signature).reshape(Npts, signature) - 0.5
        result = marked_npairs(data1, data1, rbins, period=period, 
            weights1=weights, weights2=weights, wfunc=wfunc_index)

        with pytest.raises(HalotoolsError):
            signature = _func_signature_int_from_wfunc(wfunc_index) + 1
            weights = np.random.random(Npts*signature).reshape(Npts, signature) - 0.5
            result = marked_npairs(data1, data1, rbins, period=period, 
                weights1=weights, weights2=weights, wfunc=wfunc_index)










