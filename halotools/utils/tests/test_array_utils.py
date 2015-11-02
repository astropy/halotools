#!/usr/bin/env python
import numpy as np 

from .. import array_utils

__all__ = ['test_find_idx_nearest_val']

def test_find_idx_nearest_val():

    # Create an array of randomly sorted integers
    x = np.arange(-10, 10)
    randomizer = np.random.random(len(x))
    ransort = np.argsort(randomizer)
    x = x[ransort]

    # Check that you always get an exactly matching element 
    # when the inputs are elements of x
    v = np.arange(-10, 10)
    result = []
    for elt in v:
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result.append(closest_val-elt)
    assert np.allclose(result, 0)

    # Check that you never differ by more than 0.5 when 
    # your inputs are within the range spanned by x
    Npts = 1e4
    v = np.random.uniform(x.min()-0.5, x.max()+0.5,Npts)
    result = np.empty(Npts)
    for ii, elt in enumerate(v):
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result[ii] = abs(closest_val - elt)
    assert np.all(result <= 0.5)

    # Check that values beyond the upper bound are handled correctly
    Npts = 10
    v = np.random.uniform(x.max()+10, x.max()+11,Npts)
    result = np.empty(Npts)
    for ii, elt in enumerate(v):
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result[ii] = abs(closest_val - elt)
    assert np.all(result >= 10)
    assert np.all(result <= 11)

    # Check that values beyond the lower bound are handled correctly
    v = np.random.uniform(x.min()-11, x.min()-10,Npts)
    result = np.empty(Npts)
    for ii, elt in enumerate(v):
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result[ii] = abs(closest_val - elt)
    assert np.all(result >= 10)
    assert np.all(result <= 11)

def test_convert_to_ndarray():
    """
    """
    x = 0
    xarr = array_utils.convert_to_ndarray(x)
    assert type(xarr) == np.ndarray
    assert len(xarr) == 1
    assert type(xarr[0]) == np.int64

    xarr = array_utils.convert_to_ndarray(x, dt = float)
    assert type(xarr) == np.ndarray
    assert len(xarr) == 1
    assert type(xarr[0]) == np.float64

    y = [0]
    yarr = array_utils.convert_to_ndarray(y)
    assert type(yarr) == np.ndarray
    assert len(yarr) == 1
    assert type(yarr[0]) == np.int64

    yarr = array_utils.convert_to_ndarray(y, dt = float)
    assert type(yarr) == np.ndarray
    assert len(yarr) == 1
    assert type(yarr[0]) == np.float64

    z  = None 
    zarr = array_utils.convert_to_ndarray(z)
    assert type(zarr) == np.ndarray
    assert len(zarr) == 1
    assert type(zarr[0]) == type(None)

    t = np.array(1)
    tarr  = array_utils.convert_to_ndarray(t)
    assert type(tarr) == np.ndarray
    assert len(tarr) == 1
    assert type(tarr[0]) == np.int64

    tarr  = array_utils.convert_to_ndarray(t, dt = float)
    assert type(tarr) == np.ndarray
    assert len(tarr) == 1
    assert type(tarr[0]) == np.float64

    u = np.array([1])
    uarr = array_utils.convert_to_ndarray(u)
    assert type(uarr) == np.ndarray
    assert len(uarr) == 1
    assert type(uarr[0]) == np.int64

    uarr = array_utils.convert_to_ndarray(u, dt = float)
    assert type(uarr) == np.ndarray
    assert len(uarr) == 1
    assert type(uarr[0]) == np.float64

    v = np.array('abc')
    varr = array_utils.convert_to_ndarray(v) 
    assert type(varr) == np.ndarray
    assert len(varr) == 1
    assert type(varr[0]) == np.string_

    v = 'abc'
    varr = array_utils.convert_to_ndarray(v) 
    assert type(varr) == np.ndarray
    assert len(varr) == 1
    assert type(varr[0]) == np.string_

    v = np.array(u'abc')
    varr = array_utils.convert_to_ndarray(v) 
    assert type(varr) == np.ndarray
    assert len(varr) == 1
    assert type(varr[0]) == np.unicode_

    v = u'abc'
    varr = array_utils.convert_to_ndarray(v) 
    assert type(varr) == np.ndarray
    assert len(varr) == 1
    assert type(varr[0]) == np.unicode_




