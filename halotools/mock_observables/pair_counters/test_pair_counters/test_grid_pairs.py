#!/usr/bin/env python

import numpy as np
from ..pairs import npairs as simp_npairs
from ..grid_pairs import npairs, xy_z_npairs
from time import time
import pstats, cProfile


def test_npairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

    result = npairs(data1, data1, rbins, Lbox=Lbox, period=period)
    
    test_result = simp_npairs(data1, data1, rbins, period=period)

    assert np.all(test_result==result), "pair counts are incorrect"


def test_npairs_nonperiodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

    result = npairs(data1, data1, rbins, Lbox=Lbox, period=None)
    test_result = simp_npairs(data1, data1, rbins, period=None)
    
    assert np.all(test_result==result), "pair counts are incorrect"


def test_npairs_speed():

    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.array([0.0,0.05,0.1])
    
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}".format(np.max(rbins)))

    #w/ PBCs
    start = time()
    result = npairs(data1, data1, rbins, Lbox=Lbox, period=period)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = npairs(data1, data1, rbins, Lbox=Lbox, period=None)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    
    pass


def test_xy_z_npairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    rp_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    pi_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    
    #two points separated along perpendicular direction, but have exactly 0 separation
    #in projection.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.5,0.3]])

    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert np.all(binned_result==0), "pi seperated, 0 rp seperated incorrect"
    
    #two points separated along parallel direction, but have exactly 0 separation
    #in parallel direction.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.4,0.5]])

    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert np.all(binned_result==0), "rp seperated, 0 pi seperated incorrect"
    
    #two points separated along parallel direction
    data1 = np.array([[0.5,0.5,0.5],[0.49,0.49,0.3]])
    
    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert  binned_result[0,1]==2, "pi seperated pairs incorrect"
    
    #two points separated along perpendicular direction
    data1 = np.array([[0.0,0.0,0.0],[0.2,0.0,0.01]])
    
    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert  binned_result[1,0]==2, "rp seperated pairs incorrect"




