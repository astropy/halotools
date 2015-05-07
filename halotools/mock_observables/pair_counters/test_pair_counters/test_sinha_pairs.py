#!/usr/bin/env python

#import modules
from __future__ import print_function, division
import numpy as np
from ..sinha_pairs import countpairs
from ..cpairs import npairs

__all__=['test_countpairs']

def test_countpairs():

    N = 1000
    data = np.random.uniform(0.0, 250.0, N*3).reshape(N,3)
    bins = np.linspace(0.0,100.0,10)
    
    counts1 = countpairs(data, data, bins, period=250.0)
    counts2 = npairs(data, data, bins, period=250.0)
    
    assert len(counts1)==len(counts2)
    assert np.all(counts1==counts2)
