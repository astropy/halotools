#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import sys

from halotools.utils.aggregation import *

def test_add_group_property():
    
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
    
    f = lambda x: np.mean(x['y'])
    
    result = add_group_property(data, 'avg_y', f, 'x', groups=None)

    assert 'GroupID' in result.dtype.names, 'GroupID field not present'
    
    assert len(result)==2, 'too many groups in group array, there should be 2'
    
    assert np.all(result['avg_y'] ==1.5), "group property not correctly calculated"


def test_add_members_property():
    
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
    
    f = lambda x: np.mean(x['y'])
    
    result = add_members_property(data, 'x', 'avg_y', f)

    print(result)
    
    assert len(result)==len(data), 'too rows in members output array'
    
    assert np.all(result['avg_y'] ==1.5), "group property not correctly calculated"


def test_binned_aggregation_group_property():
    
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
    
    f = lambda x: np.mean(x['y'])
    
    bins = np.arange(0,3,1)
    bins, result = binned_aggregation_group_property(data, 'x', bins, f)
    
    assert len(result)==2, 'too rows in members output array'
    
    assert np.all(result ==1.5), "group property not correctly calculated"


def test_binned_aggregation_members_property():
    
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
    
    f = lambda x: np.mean(x['y'])
    
    bins = np.arange(0,3,1)
    bins, result = binned_aggregation_members_property(data, 'x', bins, f)
    
    assert len(result)==len(data), 'too rows in members output array'
    
    assert np.all(result ==1.5), "group property not correctly calculated"


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
    
    result = group_by(data, keys=['x','y'], function=None, append_as_GroupID=True)
    #there will be a new field, GroupID, with 4 groups with tags 0,1,2,3
    assert np.all(np.unique(result['GroupID'])==[0,1,2,3])
    assert len(result)==100