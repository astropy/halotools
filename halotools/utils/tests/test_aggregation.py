#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import sys

from halotools.utils.aggregation import *

def test_group_by():
    
    x = np.empty(100)
    y = np.empty(100)
    
    #the first half is type 1 and the second type 2
    x[0:50]=1
    x[50:]=2
    #the 1st and 3rd quarters are type 1, and the 2nd and 4th are type 2 
    y[0:25]=1
    y[25:50]=2
    y[50:75]=1
    y[75:]=2
    
    dtype=np.dtype([('x',x.dtype),('y',y.dtype)])
    data=np.empty((len(x),),dtype=dtype)
    data['x']=x
    data['y']=y
    
    result = group_by(data, keys=['x'], function=None, append_as_GroupID=False)
    #there will be 2 groups with tags 0,1
    assert np.all(np.unique(result)==[0,1])
    assert len(result)==100
    
    result = group_by(data, keys=['x','y'], function=None, append_as_GroupID=False)
    #there will be 4 groups with tags 0,1,2,3
    assert np.all(np.unique(result)==[0,1,2,3])
    assert len(result)==100