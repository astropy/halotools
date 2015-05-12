#!/usr/bin/env python

"""
#import modules
from __future__ import division
from ..kdtrees.ckdtree import cKDTree
#import simple pair counter to compare results with ckdtree pair counter
from halotools.mock_observables.pair_counters.pairs import npairs
from halotools.mock_observables.pair_counters.pairs import pairs
from halotools.mock_observables.pair_counters.pairs import wnpairs
from halotools.mock_observables.pair_counters.cpairs import pairwise_distances
#other modules
import numpy as np
import sys
"""

__all__=['test_initialization', 'test_count_neighbors', 'test_count_neighbors_periodic',\
         'test_count_neighbors_approximation', 'test_query_pairs',\
         'test_query_pairs_periodic', 'test_query_ball_tree',\
         'test_query_ball_tree_periodic', 'test_query_ball_point',\
         'test_query_ball_point_periodic', 'test_query_ball_point_wcounts',\
         'test_query', 'test_query_periodic', 'test_wcount_neighbors_periodic',\
         'test_wcount_neighbors_large', 'test_wcount_neighbors_double_weights',\
         'test_wcount_neighbors_double_weights_functionality',\
         'test_wcount_neighbors_custom_double_weights_functionality',\
         'test_wcount_neighbors_custom_2D_double_weights_functionality',\
         'test_wcount_neighbors_custom_2D_double_weights_pbcs',\
         'test_sparse_distance_matrix']

"""
This script contains code to test the functionality of ckdtree.pyx 
"""

"""
##########################################################################################
#tests for initialization
##########################################################################################
def test_initialization():
    
    data = np.random.random((100,3))
    
    tree = cKDTree(data)
    
    assert tree.n == 100, 'number of points is not 10'
    assert tree.m == 3, 'dimension of points is not 3'
    assert np.all(tree.data == data), 'data indexing changed during intialization?'
    assert np.all(tree.mins == np.amin(data,axis=0)), 'mins are off'
    assert np.all(tree.maxes == np.amax(data,axis=0)), 'maxes are off'
    assert tree.leafsize==10, 'default leafsize is not 10'


##########################################################################################
#tests for count_neighbors
##########################################################################################
def test_count_neighbors():

    data_1 = np.random.random((100,3))
    data_2 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    n1 = npairs(data_1,data_2,0.25)[0]
    
    n2 = tree_1.count_neighbors(tree_2,0.25)
    
    assert n1==n2, 'tree calc did not find same number of pairs'


def test_count_neighbors_periodic():

    data_1 = np.random.random((100,3))
    data_2 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    period = np.array([1,1,1])
    
    n1 = npairs(data_1,data_2,0.25, period=period)[0]
    
    n2 = tree_1.count_neighbors(tree_2,0.25, period=period)
    
    assert n1==n2, 'tree calc did not find same number of pairs'


def test_count_neighbors_approximation():
    import time
    data_1 = np.random.random((10000,3))
    data_2 = np.random.random((1000,3))
    
    r_bins = np.logspace(-2,0,10)
    r_bins = 0.5
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    eps=0.0
    
    n1 = npairs(data_1,data_2,r_bins)
    start = time.time()
    n2 = tree_1.count_neighbors(tree_2,r_bins,eps=eps)
    print time.time()-start
    
    #print n1
    #print n2
    #assert True==False


##########################################################################################
#tests for query pairs
##########################################################################################
def test_query_pairs():
    data_1 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    
    p1 = pairs(data_1, 0.25)
    p2 = tree_1.query_pairs(0.25)
    
    assert p1==p2, 'not all pairs found'


def test_query_pairs_periodic():
    data_1 = np.random.random((100,3))
    
    period = np.array([1,1,1])
    
    tree_1 = cKDTree(data_1)
    
    p1 = pairs(data_1, 0.25, period=period)
    p2 = tree_1.query_pairs(0.25, period=period)
    
    print(p1.difference(p2))
    assert p1==p2, 'not all pairs found'


##########################################################################################
#tests for query_ball_tree
##########################################################################################
def test_query_ball_tree():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    data_2 = np.random.random((100,3))
    tree_2 = cKDTree(data_2)
    
    n1 = npairs(data_1, data_2, 0.4)
    n2 = tree_1.query_ball_tree(tree_2, 0.4)
    n2 = sum(len(x) for x in n2) #number of pairs found
    
    assert n1==n2, 'inconsistent number found'


def test_query_ball_tree_periodic():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    data_2 = np.random.random((100,3))
    tree_2 = cKDTree(data_2)
    
    period = np.array([1,1,1])
    
    n1 = npairs(data_1, data_2, 0.4, period=period)
    n2 = tree_1.query_ball_tree(tree_2, 0.4, period=period)
    n2 = sum(len(x) for x in n2) #number of pairs found
    
    assert n1==n2, 'inconsistent number found'


##########################################################################################
#tests for query_ball_point
##########################################################################################
def test_query_ball_point():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.5,0.5,0.5])
    
    n1 = npairs(data_1,x,0.5)[0]
    n2 = len(tree_1.query_ball_point(x,0.5))
    
    assert n1==n2, 'inconsistent number of points found...'


def test_query_ball_point_periodic():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.1,0.1,0.1])
    
    period = np.array([1,1,1])
    
    n1 = npairs(data_1,x,0.4, period=period)[0]
    n2 = len(tree_1.query_ball_point(x,0.4, period=period))
    
    assert n1==n2, 'inconsistent number of points found...'


def test_query_ball_point_wcounts():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    #create random coordinates
    N1 = 1000
    data_1 = np.random.random((N1,3))
    p=np.random.random((3,))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    
    #define random weights for test data set 2
    weights1 = np.random.random((N1,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, p, 1.0, weights1=weights1)[0]
    n1 = tree_1.query_ball_point_wcounts(p, 1.0, weights=weights1)
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


##########################################################################################
#tests for query
##########################################################################################
def test_query(): #not a very good test...
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.5,0.5,0.5])
    
    ps = tree_1.query(x,10)[0]
    print ps
    assert len(ps)==10, 'inconsistent number of points found...'


def test_query_periodic(): #not a very good test...
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.5,0.5,0.5])
    
    period = np.array([1,1,1])
    
    ps = tree_1.query(x,10,period=period)[0]
    print ps
    assert len(ps)==10, 'inconsistent number of points found...'


##########################################################################################
#tests for wcount_neighbors
##########################################################################################
def test_wcount_neighbors_periodic():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    N1 = 100
    N2 = 100
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    period = np.array([1,1,1])
    weights = np.zeros((N2,))+0.1
    
    n0 = wnpairs(data_1, data_2, 0.1, period=period, weights2=weights)[0]
    n1 = tree_1.wcount_neighbors(tree_2,0.1, period=period) #no weights
    n2 = tree_1.wcount_neighbors(tree_2,0.1, period=period, oweights=weights) #constant weights
    n3 = tree_1.count_neighbors(tree_2,0.1, period=period) #non-weighted function
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n2))
    assert np.fabs(n2-n3*0.1)/n0 < 10.0 * ep, 'error in weighted counts.'
    assert np.fabs(n0-n2)/n0 < 10.0 * ep, 'error in weighted.'
    
    #define random weights for test data set 2
    weights = np.random.random((len(data_2),))
    
    n0 = wnpairs(data_1, data_2, 0.25, period=period, weights2=weights)[0]
    n2 = tree_1.wcount_neighbors(tree_2,0.25, period=period, oweights=weights) #constant weights
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n2))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n2)/n0,ep))
    assert np.fabs(n0-n2)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


def test_wcount_neighbors_large():
    return 0 #skip this as it takes some time!
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    #create random coordinates
    N1 = 10000
    N2 = 10000
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights = np.random.random((N2,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, data_2, 1.0, weights2=weights)[0]
    n1 = tree_1.wcount_neighbors(tree_2,1.0, oweights=weights)
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


def test_wcount_neighbors_double_weights():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    #create random coordinates
    N1 = 100
    N2 = 100
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights1 = np.random.random((N1,))
    weights2 = np.random.random((N2,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, data_2, 1.0, weights1=weights1, weights2=weights2)[0]
    n1 = tree_1.wcount_neighbors(tree_2,1.0, sweights=weights1, oweights=weights2)
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


def test_wcount_neighbors_double_weights_functionality():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])
    
    #user defined function
    from ..kdtrees.ckdtree import Function
    class MyFunction(Function):
        def evaluate(self, x, y, a, b):
            return x*y

    #create random coordinates
    N1 = 100
    N2 = 100
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights1 = np.random.random((N1,))
    weights2 = np.random.random((N2,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, data_2, 1.0, weights1=weights1, weights2=weights2)[0]
    n1 = tree_1.wcount_neighbors(tree_2,1.0, sweights=weights1, oweights=weights2, w=MyFunction())
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


##########################################################################################
#tests for wcount_neighbors_custom
##########################################################################################
def test_wcount_neighbors_custom_double_weights_functionality():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])
    
    #user defined function
    from ..kdtrees.ckdtree import Function
    class MyFunction(Function):
        def evaluate(self, x, y, a, b):
            return x*y

    #create random coordinates
    N1 = 1000
    N2 = 1000
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    r = np.arange(0.1,1,0.1)
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights1 = np.random.random((N1,))
    weights2 = np.random.random((N2,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, data_2, r, weights1=weights1, weights2=weights2)
    n1 = tree_1.wcount_neighbors_custom(tree_2,r, sweights=weights1, oweights=weights2, w=MyFunction())
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    #sum to compare to brute force method.
    n1 = np.sum(n1, axis=0)
    
    print(n1)
    print(n0)
    assert np.all(np.fabs(n0-n1)/n0 < 10.0 * ep), 'weights are being handeled incorrectly'


##########################################################################################
#tests for wcount_neighbors_custom_2D
##########################################################################################
def test_wcount_neighbors_custom_2D_double_weights_functionality():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])
    
    #user defined function
    from ..kdtrees.ckdtree import Function
    class MyFunction(Function):
        def evaluate(self, x, y, a, b):
            if a==0: return 1
            elif (x==y) & (x==a): return 0.0
            elif x==y: return 1.0
            elif x!=y: return 0.5

    #create random coordinates
    N1 = 100
    N2 = 1000
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    r = np.arange(0.1,0.5,0.1)
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights1 = np.random.random_integers(1,100,size=N1)
    weights2 = np.random.random_integers(1,100,size=N2)
    wdim = 101
    
    #calculate weighted sums
    n0 = npairs(data_1, data_2, r)
    n1 = tree_1.wcount_neighbors_custom_2D(tree_2,r, sweights=weights1, oweights=weights2, wdim=wdim)[0]
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print(n0,n1)
    print(np.fabs(n0-n1)/n0 < 10.0 * ep)
    assert np.all(np.fabs(n0-n1)/n0 < 10.0 * ep), 'weights are being handeled incorrectly'


def test_wcount_neighbors_custom_2D_double_weights_pbcs():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])
    
    #user defined function
    from ..kdtrees.ckdtree import Function
    class MyFunction(Function):
        def evaluate(self, x, y, a, b):
            if a==0: return 1
            elif (x==y) & (x==a): return 0.0
            elif x==y: return 1.0
            elif x!=y: return 0.5

    #create random coordinates
    N1 = 100
    N2 = 1000
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    period = np.array([1,1,1])
    r = np.arange(0.1,0.5,0.1)
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights1 = np.random.random_integers(1,100,size=N1)
    weights2 = np.random.random_integers(1,100,size=N2)
    wdim = 101
    
    #calculate weighted sums
    n0 = npairs(data_1, data_2, r, period=period)
    n1 = tree_1.wcount_neighbors_custom_2D(tree_2,r, sweights=weights1, oweights=weights2, wdim=wdim, period=period)[0]
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print(n0,n1)
    print(np.fabs(n0-n1)/n0 < 10.0 * ep)
    assert np.all(np.fabs(n0-n1)/n0 < 10.0 * ep), 'weights are being handeled incorrectly'


##########################################################################################
#tests for sparse_distance_matrix
##########################################################################################
def test_sparse_distance_matrix():

    data_1 = np.random.random((1000,3))
    data_2 = np.random.random((1000,3))
    
    period = np.array([1,1,1])
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    result_1 = tree_1.sparse_distance_matrix(tree_1, 0.1, period=period)
    
    from scipy.sparse import coo_matrix
    result_2 = pairwise_distances(data_1, period=period, max_distance=0.1)
    result_2 = coo_matrix(result_2)
    
    diff = (result_1-result_2)
    epsilon = np.float64(sys.float_info[8])
    print(epsilon)
    print(np.abs(diff)<epsilon)
    
    assert (np.abs(diff)>epsilon).nnz==0


    """