#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest 
#load comparison simple pair counters
from ..pairs import npairs as simp_npairs
from ..pairs import wnpairs as simp_wnpairs
#load rect_cuboid_pairs pair counters
from ..rect_cuboid_pairs import npairs, wnpairs, jnpairs
from ..rect_cuboid_pairs import xy_z_npairs, xy_z_wnpairs, xy_z_jnpairs
from ..rect_cuboid_pairs import s_mu_npairs


np.random.seed(1)

@pytest.mark.slow
def test_npairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

    result = npairs(data1, data1, rbins, Lbox=Lbox, period=period, verbose=True, num_threads=1)
    
    test_result = simp_npairs(data1, data1, rbins, period=period)

    print(result)
    print(test_result)
    assert np.all(test_result==result), "pair counts are incorrect"


@pytest.mark.slow
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


@pytest.mark.slow
def test_xy_z_npairs_periodic():
    
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    rp_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    pi_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    
    #two points separated along perpendicular direction, but have exactly 0 separation
    #in projection.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.5,0.3]])

    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    print(result)
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


@pytest.mark.slow
def test_xy_z_npairs_nonperiodic():
    
    Lbox = [1.0,1.0,1.0]
    period = None
    
    rp_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    pi_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    
    #two points separated along perpendicular direction, but have exactly 0 separation
    #in projection.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.5,0.3]])

    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    print(result)
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


@pytest.mark.slow
def test_s_mu_npairs_periodic():
    
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    s_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    N_mu_bins=100
    mu_bins = np.linspace(0,1.0,N_mu_bins)
    
    Npts = 1e3
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    result = s_mu_npairs(data1, data1, s_bins, mu_bins, period=period)
    
    assert np.shape(result)==(6,N_mu_bins), "result has the wrong shape"
    
    result = np.diff(np.diff(result,axis=1),axis=0)
    
    xi = np.sum(result,axis=1)
    
    comp_result = npairs(data1, data1, s_bins, period=period)
    comp_result = np.diff(comp_result)
    
    assert np.all(xi==comp_result), "pair counts don't match simple pair counts"


@pytest.mark.slow
def test_s_mu_npairs_nonperiodic():
    
    Lbox = [1.0,1.0,1.0]
    period = None
    
    s_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    N_mu_bins=100
    mu_bins = np.linspace(0,1.0,N_mu_bins)
    
    Npts = 1e3
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    result = s_mu_npairs(data1, data1, s_bins, mu_bins, period=None, Lbox=Lbox)
    
    assert np.shape(result)==(6,N_mu_bins), "result has the wrong shape"
    
    result = np.diff(np.diff(result,axis=1),axis=0)
    
    xi = np.sum(result,axis=1)
    
    comp_result = npairs(data1, data1, s_bins, period=None)
    comp_result = np.diff(comp_result)
    
    print(comp_result, xi)
    assert np.all(xi==comp_result), "pair counts don't match simple pair counts"
    


@pytest.mark.slow
def test_wnpairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

    result = wnpairs(data1, data1, rbins, Lbox=Lbox, period=period, weights1=weights1, weights2=weights1)
    
    test_result = simp_wnpairs(data1, data1, rbins, period=period, weights1=weights1, weights2=weights1)

    print(test_result,result)
    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


@pytest.mark.slow
def test_wnpairs_nonperiodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    
    rbins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

    result = wnpairs(data1, data1, rbins, Lbox=Lbox, period=None, weights1=weights1, weights2=weights1)
    
    test_result = simp_wnpairs(data1, data1, rbins, period=None, weights1=weights1, weights2=weights1)

    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"


@pytest.mark.slow
def test_xy_z_wnpairs_periodic():
    
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    weights1 = np.ones(2)
    rp_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    pi_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    
    #two points separated along perpendicular direction, but have exactly 0 separation
    #in projection.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.5,0.3]])

    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    print(result)
    assert np.all(binned_result==0), "pi seperated, 0 rp seperated incorrect"
    
    #two points separated along parallel direction, but have exactly 0 separation
    #in parallel direction.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.4,0.5]])

    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert np.all(binned_result==0), "rp seperated, 0 pi seperated incorrect"
    
    #two points separated along parallel direction
    data1 = np.array([[0.5,0.5,0.5],[0.49,0.49,0.3]])
    
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert  binned_result[0,1]==2, "pi seperated pairs incorrect"
    
    #two points separated along perpendicular direction
    data1 = np.array([[0.0,0.0,0.0],[0.2,0.0,0.01]])
    
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert  binned_result[1,0]==2, "rp seperated pairs incorrect"


@pytest.mark.slow
def test_xy_z_wnpairs_nonperiodic():
    
    Lbox = [1.0,1.0,1.0]
    period = None
    
    weights1 = np.ones(2)
    rp_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    pi_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    
    #two points separated along perpendicular direction, but have exactly 0 separation
    #in projection.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.5,0.3]])

    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    print(result)
    assert np.all(binned_result==0), "pi seperated, 0 rp seperated incorrect"
    
    #two points separated along parallel direction, but have exactly 0 separation
    #in parallel direction.  should return 0 pairs in bins
    data1 = np.array([[0.5,0.5,0.5],[0.5,0.4,0.5]])

    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert np.all(binned_result==0), "rp seperated, 0 pi seperated incorrect"
    
    #two points separated along parallel direction
    data1 = np.array([[0.5,0.5,0.5],[0.49,0.49,0.3]])
    
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert  binned_result[0,1]==2, "pi seperated pairs incorrect"
    
    #two points separated along perpendicular direction
    data1 = np.array([[0.0,0.0,0.0],[0.2,0.0,0.01]])
    
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    assert  binned_result[1,0]==2, "rp seperated pairs incorrect"


@pytest.mark.slow
def test_jnpairs_periodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    jtags1 = np.sort(np.random.random_integers(1,10,size=Npts))
    
    rbins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

    result = jnpairs(data1, data1, rbins, Lbox=Lbox, period=period,\
                     jtags1=jtags1, jtags2=jtags1, N_samples=10,\
                     weights1=weights1, weights2=weights1)
    
    print(result)

    assert result.ndim==2, 'result is the wrong dimension'
    assert np.shape(result)==(11,6), 'result is the wrong shape'


@pytest.mark.slow
def test_jnpairs_nonperiodic():
    
    Npts = 1e3
    Lbox = [1.0,1.0,1.0]
    period = None
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    jtags1 = np.sort(np.random.random_integers(1,10,size=Npts))
    
    rbins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

    result = jnpairs(data1, data1, rbins, Lbox=Lbox, period=period,\
                     jtags1=jtags1, jtags2=jtags1, N_samples=10,\
                     weights1=weights1, weights2=weights1)
    
    print(result)

    assert result.ndim==2, 'result is the wrong dimension'
    assert np.shape(result)==(11,6), 'result is the wrong shape'


@pytest.mark.slow
def test_xy_z_jnpairs_periodic():
    
    Npts=100
    Lbox = [1.0,1.0,1.0]
    period = np.array(Lbox)
    
    rp_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    pi_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    jtags1 = np.sort(np.random.random_integers(1, 10, size=Npts))

    result = xy_z_jnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period, jtags1=jtags1, jtags2=jtags1, N_samples=10)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    result_compare = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    
    print(np.shape(result))
    assert np.shape(result)==(11,6,6), "shape xy_z jackknife pair counts of result is incorrect"
    
    assert np.all(result[0]==result_compare), "shape xy_z jackknife pair counts of result is incorrect"


@pytest.mark.slow
def test_xy_z_jnpairs_nonperiodic():
    
    Npts=100
    Lbox = [1.0,1.0,1.0]
    period = None
    
    rp_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    pi_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5])
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    jtags1 = np.sort(np.random.random_integers(1, 10, size=Npts))

    result = xy_z_jnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period, jtags1=jtags1, jtags2=jtags1, N_samples=10)
    binned_result = np.diff(np.diff(result,axis=1),axis=0)
    
    result_compare = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    
    print(np.shape(result))
    assert np.shape(result)==(11,6,6), "shape xy_z jackknife pair counts of result is incorrect"
    
    assert np.all(result[0]==result_compare), "shape xy_z jackknife pair counts of result is incorrect"
    
    
    