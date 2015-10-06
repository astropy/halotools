#!/usr/bin/env python

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from astropy.table import Table
import sys

from .. aggregation import *

def test_add_group_property():
    
    x = np.empty(100)
    y = np.empty(100)
    
    #the first half is type 1 and the second type 2, e.g. group ids
    x[0:50]=1
    x[50:]=2
    #the 1st and 3rd quarters are type 1, and the 2nd and 4th are type 2 
    y[0:25]=1
    y[25:50]=2
    y[50:75]=1
    y[75:]=2
    
    #define groups array
    groups = Table([np.zeros(2)], names=['group_id'], dtype=['>i8'])
    groups['group_id'][0] = 1
    groups['group_id'][1] = 2
    
    members = Table([x,y], names=['group_id','y'], dtype=['>i8','>f8'])

    f = lambda x: np.mean(x['y'])
    
    groups = add_group_property(members, 'group_id', f, groups, 'avg_y')

    assert len(groups)==2, 'too many groups in group array, there should be 2'
    
    assert np.all(groups['avg_y'] ==1.5), "group property not correctly calculated"


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
    
    data = Table([x,y], names=['x','y'], dtype=[x.dtype.str,y.dtype.str])
    
    f = lambda x: np.mean(x['y'])
    
    result = add_members_property(data, 'x', 'avg_y', f)
    
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
    
    data = Table([x,y], names=['x','y'], dtype=[x.dtype.str,y.dtype.str])
    
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
    
    data = Table([x,y], names=['x','y'], dtype=[x.dtype.str,y.dtype.str])
    
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
    
    data = Table([x,y], names=['x','y'], dtype=[x.dtype.str,y.dtype.str])
    
    result = group_by(data, keys=['x'], function=None, append_id_field=None)
    #there will be 2 groups with tags 0,1
    assert np.all(np.unique(result)==[0,1])
    assert len(result)==100
    
    result = group_by(data, keys=['x','y'], function=None, append_id_field=None)
    #there will be 4 groups with tags 0,1,2,3
    assert np.all(np.unique(result)==[0,1,2,3])
    assert len(result)==100
    
    result = group_by(data, keys=['x','y'], function=None, append_id_field='group_id')
    #there will be a new field, GroupID, with 4 groups with tags 0,1,2,3
    assert np.all(np.unique(result['group_id'])==[0,1,2,3])
    assert len(result)==100