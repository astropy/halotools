
from __future__ import print_function, division
import numpy as np
import halotools.mock_observables.pair_counters.sinha_pairs as sinha_pairs
import halotools.mock_observables.pair_counters.cpairs as cpairs

def test_countpairs():

    N = 1000
    data = np.random.uniform(0.0, 250.0, N*3).reshape(N,3)
    bins = np.linspace(0.0,100.0,10)
    
    counts1 = sinha_pairs.countpairs(data,data,bins,period=250.0)
    counts2 = cpairs.npairs(data, data, bins, period=250.0)
    
    print(counts1,counts2, len(counts1), len(counts2))
    print(counts1-counts2)
    assert np.all(counts1==counts2)
