#!/usr/bin/env python

#import modules
from __future__ import print_function, division
import numpy as np
from ..sinha_pairs import npairs
from ..cpairs import npairs as npairs_compare

__all__=['test_npairs_auto', 'test_npairs_cross', 'test_bins']

def test_npairs_auto():

    N = 1000
    data = np.random.uniform(0.0, 250.0, N*3).reshape(N,3)
    bins = np.linspace(0.0,100.0,10)
    
    counts1 = npairs(data, data, bins, period=250.0)
    counts2 = npairs_compare(data, data, bins, period=250.0)
    
    assert len(counts1)==len(counts2)
    assert np.all(counts1==counts2)


def test_npairs_cross():

    N = 1000
    data1 = np.random.uniform(0.0, 250.0, N*3).reshape(N,3)
    data2 = np.random.uniform(0.0, 250.0, N*3).reshape(N,3)
    bins = np.linspace(0.0,100.0,10)
    
    counts1 = npairs(data1, data2, bins, period=250.0)
    counts2 = npairs_compare(data1, data2, bins, period=250.0)
    
    assert len(counts1)==len(counts2)
    assert np.all(counts1==counts2)


def test_bins():

    N = 1000
    data = np.random.uniform(0.0, 250.0, N*3).reshape(N,3)
    bins = np.linspace(0.0,125.0,10)
    
    counts1 = npairs(data, data, bins, period=250.0)
    counts2 = npairs_compare(data, data, bins, period=250.0)
    
    assert len(counts1)==len(counts2)
    assert np.all(counts1==counts2)
