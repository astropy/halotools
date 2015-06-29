#!/usr/bin/env python

import numpy as np
#load comparison simple pair counters
from .. FoF_pairs import fof_pairs


def test_npairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    r_max=0.1
    
    d, i, j = fof_pairs(data1, data1, r_max, period=period)
    
    print(d,i,j)
    assert True==False
    