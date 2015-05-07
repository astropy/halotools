#!/usr/bin/env python

#Duncan Campbell
#September 2, 2014
#Yale University
#Compare various methods/codes for pair counting.

from halotools.mock_observables.pairs import npairs as npairs
from halotools.mock_observables.cpairs import npairs as cnpairs
from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree as cKDTree_custom
from scipy.spatial import cKDTree as cKDTree_scipy
import numpy as np
import time

def main():
    N1=10**4
    N2=10**4
    sample1 = np.random.random((N1,3))
    sample2 = np.random.random((N2,3))
    rbins = np.logspace(-3,-0.31,50)
    period = np.array([1,1])
    
    '''
    #brute force python
    start = time.time()
    result = npairs(sample,sample,rbins,period=period)
    dt = time.time()-start
    print dt
    '''
    '''
    #brute force cython
    start = time.time()
    result = cnpairs(sample1,sample2,rbins,period=period)
    dt = time.time()-start
    print dt
    '''
    
    #ckdtree
    tree1 = cKDTree_scipy(sample1,leafsize=100)
    tree2 = cKDTree_scipy(sample2,leafsize=1000)
    start = time.time()
    result = tree1.count_neighbors(tree2,rbins)
    dt = time.time()-start
    print dt
    
    #ckdtree
    tree1 = cKDTree_custom(sample1,leafsize=100)
    tree2 = cKDTree_custom(sample2,leafsize=1000)
    start = time.time()
    result = tree1.count_neighbors(tree2,rbins)
    dt = time.time()-start
    print dt


if __name__ == '__main__':
    main()
