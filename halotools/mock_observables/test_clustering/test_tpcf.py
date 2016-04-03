#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys
from multiprocessing import cpu_count 

from ..tpcf import tpcf
from ...custom_exceptions import *

import pytest
slow = pytest.mark.slow

__all__=['test_tpcf_auto', 'test_tpcf_cross', 'test_tpcf_estimators',\
         'test_tpcf_sample_size_limit',\
         'test_tpcf_randoms', 'test_tpcf_period_API', 'test_tpcf_cross_consistency_w_auto']

"""
Note that these are almost all unit-tests.  Non tirival tests are a little heard to think
of here.
"""

@slow
def test_tpcf_auto():
    """
    test the tpcf auto-correlation functionality
    """

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    #with randoms
    result = tpcf(sample1, rbins, sample2 = None, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural', 
                  approx_cell1_size = [rmax, rmax, rmax], 
                  approx_cellran_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."

    #with out randoms
    result = tpcf(sample1, rbins, sample2 = None, 
                  randoms=None, period = period, 
                  max_sample_size=int(1e4), estimator='Natural', 
                  approx_cell1_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."


@slow
def test_tpcf_cross():
    """
    test the tpcf cross-correlation functionality
    """

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    #with randoms
    result = tpcf(sample1, rbins, sample2 = sample2, 
                  randoms=randoms, period = None, 
                  max_sample_size=int(1e4), estimator='Natural', do_auto=False, 
                  approx_cell1_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."

    #with out randoms
    result = tpcf(sample1, rbins, sample2 = sample2, 
                  randoms=None, period = period, 
                  max_sample_size=int(1e4), estimator='Natural', do_auto=False, 
                  approx_cell1_size = [rmax, rmax, rmax])
    assert result.ndim == 1, "More than one correlation function returned erroneously."

@slow
def test_tpcf_estimators():
    """
    test the tpcf different estimators functionality
    """

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Davis-Peebles', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_3 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Hewett', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_4 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Hamilton', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    result_5 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Landy-Szalay', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])


    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of  correlation functions returned erroneously."
    assert len(result_3)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_4)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_5)==3, "wrong number of correlation functions returned erroneously."

@slow
def test_tpcf_sample_size_limit():
    """
    test the tpcf sample size limit functionality functionality
    """

    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((1000,3))
    randoms = np.random.random((1000,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e2), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax])

    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."

@slow
def test_tpcf_randoms():
    """
    test the tpcf possible randoms + PBCs combinations
    """

    sample1 = np.random.random((100,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    #No PBCs w/ randoms
    result_1 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = None, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    #PBCs w/o randoms
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=None, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])
    #PBCs w/ randoms
    result_3 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax], 
                    approx_cellran_size = [rmax, rmax, rmax])

    #No PBCs and no randoms should throw an error.
    try:
        tpcf(sample1, rbins, sample2 = sample2, 
             randoms=None, period = None, 
             max_sample_size=int(1e4), estimator='Natural', 
             approx_cell1_size = [rmax, rmax, rmax], 
             approx_cellran_size = [rmax, rmax, rmax])
    except HalotoolsError:
        pass

    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_3)==3, "wrong number of correlation functions returned erroneously."

@slow
def test_tpcf_period_API():
    """
    test the tpcf period API functionality.
    """

    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    result_1 = tpcf(sample1, rbins, sample2 = sample2,
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax])

    period = 1.0
    result_2 = tpcf(sample1, rbins, sample2 = sample2, 
                    randoms=randoms, period = period, 
                    max_sample_size=int(1e4), estimator='Natural', 
                    approx_cell1_size = [rmax, rmax, rmax])

    #should throw an error.  period must be positive!
    period = np.array([1.0,1.0,-1.0])
    try:
        tpcf(sample1, rbins, sample2 = sample2, 
             randoms=randoms, period = period, 
             max_sample_size=int(1e4), estimator='Natural', 
             approx_cell1_size = [rmax, rmax, rmax])
    except HalotoolsError:
        pass

    assert len(result_1)==3, "wrong number of correlation functions returned erroneously."
    assert len(result_2)==3, "wrong number of correlation functions returned erroneously."


@slow
def test_tpcf_cross_consistency_w_auto():
    """
    test the tpcf cross-correlation mode consistency with auto-correlation mode
    """

    sample1 = np.random.random((200,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((300,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    #with out randoms
    result1 = tpcf(sample1, rbins, sample2 = None, 
                   randoms=None, period = period, 
                   max_sample_size=int(1e4), estimator='Natural', 
                   approx_cell1_size = [rmax, rmax, rmax])

    result2 = tpcf(sample2, rbins, sample2 = None, 
                   randoms=None, period = period, 
                   max_sample_size=int(1e4), estimator='Natural', 
                   approx_cell1_size = [rmax, rmax, rmax])

    result1_p, result12, result2_p = tpcf(sample1, rbins, sample2 = sample2, 
                                          randoms=None, period = period, 
                                          max_sample_size=int(1e4),
                                          estimator='Natural', 
                                          approx_cell1_size=[rmax, rmax, rmax])

    assert np.allclose(result1,result1_p), "cross mode and auto mode are not the same"
    assert np.allclose(result2,result2_p), "cross mode and auto mode are not the same"

    #with randoms
    result1 = tpcf(sample1, rbins, sample2 = None, 
                   randoms=randoms, period = period, 
                   max_sample_size=int(1e4), estimator='Natural', 
                   approx_cell1_size = [rmax, rmax, rmax])

    result2 = tpcf(sample2, rbins, sample2 = None, 
                   randoms=randoms, period = period, 
                   max_sample_size=int(1e4), estimator='Natural', 
                   approx_cell1_size = [rmax, rmax, rmax])

    result1_p, result12, result2_p = tpcf(sample1, rbins, sample2 = sample2, 
                                          randoms=randoms, period = period, 
                                          max_sample_size=int(1e4),
                                          estimator='Natural', 
                                          approx_cell1_size=[rmax, rmax, rmax])

    assert np.allclose(result1,result1_p), "cross mode and auto mode are not the same"
    assert np.allclose(result2,result2_p), "cross mode and auto mode are not the same"


def test_RR_precomputed_exception_handling1():

    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    RR_precomputed = rmax
    with pytest.raises(HalotoolsError) as err:
        result_1 = tpcf(sample1, rbins, sample2 = sample2,
            randoms=randoms, period = period, 
            max_sample_size=int(1e4), estimator='Natural', 
            approx_cell1_size = [rmax, rmax, rmax], 
            RR_precomputed = RR_precomputed)
    substr = "``RR_precomputed`` and ``NR_precomputed`` arguments, or neither\n"
    assert substr in err.value.args[0]


def test_RR_precomputed_exception_handling2():

    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    RR_precomputed = rbins[:-2]
    NR_precomputed = randoms.shape[0]
    with pytest.raises(HalotoolsError) as err:
        result_1 = tpcf(sample1, rbins, sample2 = sample2,
            randoms=randoms, period = period, 
            max_sample_size=int(1e4), estimator='Natural', 
            approx_cell1_size = [rmax, rmax, rmax], 
            RR_precomputed = RR_precomputed, NR_precomputed = NR_precomputed)
    substr = "\nLength of ``RR_precomputed`` must match length of ``rbins``\n"
    assert substr in err.value.args[0]


def test_RR_precomputed_exception_handling3():

    sample1 = np.random.random((1000,3))
    sample2 = np.random.random((100,3))
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    RR_precomputed = rbins[:-1]
    NR_precomputed = 5
    with pytest.raises(HalotoolsError) as err:
        result_1 = tpcf(sample1, rbins, sample2 = sample2,
            randoms=randoms, period = period, 
            max_sample_size=int(1e4), estimator='Natural', 
            approx_cell1_size = [rmax, rmax, rmax], 
            RR_precomputed = RR_precomputed, NR_precomputed = NR_precomputed)
    substr = "the value of NR_precomputed must agree with the number of randoms"
    assert substr in err.value.args[0]


@slow
def test_RR_precomputed_natural_estimator_auto():
    """ Strategy here is as follows. First, we adopt the same setup 
    with randomly generated points as used in the rest of the test suite. 
    First, we just compute the tpcf in the normal way. 
    Then we break apart the tpcf innards so that we can 
    compute RR in the exact same way that it is computed within tpcf. 
    We will then pass in this RR using the RR_precomputed keyword, 
    and verify that the tpcf computed in this second way gives 
    exactly the same results as if we did not pre-compute RR.

    """
    sample1 = np.random.random((1000,3))
    sample2 = sample1
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    approx_cell1_size = [rmax, rmax, rmax]
    approx_cell2_size = approx_cell1_size
    approx_cellran_size = [rmax, rmax, rmax]

    normal_result = tpcf(
        sample1, rbins, sample2 = sample2, 
        randoms=randoms, period = period, 
        max_sample_size=int(1e4), estimator='Natural', 
        approx_cell1_size=approx_cell1_size, 
        approx_cellran_size=approx_cellran_size)


    # The following quantities are computed inside the 
    # tpcf namespace. We reproduce them here because they are 
    # necessary inputs to the _random_counts and _pair_counts 
    # functions called by tpcf 
    _sample1_is_sample2 = True
    PBCs = True
    num_threads = cpu_count()
    do_DD, do_DR, do_RR = True, True, True  
    do_auto, do_cross = True, False      

    from ..tpcf import _random_counts, _pair_counts

    #count data pairs
    D1D1,D1D2,D2D2 = _pair_counts(
        sample1, sample2, rbins, period,
        num_threads, do_auto, do_cross, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size)

    #count random pairs
    D1R, D2R, RR = _random_counts(
        sample1, sample2, randoms, rbins, period,
        PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size, approx_cellran_size)

    N1 = len(sample1)
    NR = len(randoms)

    factor = N1*N1/(NR*NR)
    mult = lambda x,y: x*y
    xi_11 = mult(1.0/factor,D1D1/RR) - 1.0

    # The following assertion implies that the RR 
    # computed within this testing namespace is the same RR 
    # as computed in the tpcf namespace
    assert np.all(xi_11 == normal_result)

    # Now we will pass in the above RR as an argument 
    # and verify that we get an identical tpcf 
    result_with_RR_precomputed = tpcf(
        sample1, rbins, sample2 = sample2, 
        randoms=randoms, period = period, 
        max_sample_size=int(1e4), estimator='Natural', 
        approx_cell1_size=approx_cell1_size, 
        approx_cellran_size=approx_cellran_size, 
        RR_precomputed = RR, 
        NR_precomputed = NR)

    assert np.all(result_with_RR_precomputed == normal_result)


@slow
def test_RR_precomputed_Landy_Szalay_estimator_auto():
    """ Strategy here is as follows. First, we adopt the same setup 
    with randomly generated points as used in the rest of the test suite. 
    First, we just compute the tpcf in the normal way. 
    Then we break apart the tpcf innards so that we can 
    compute RR in the exact same way that it is computed within tpcf. 
    We will then pass in this RR using the RR_precomputed keyword, 
    and verify that the tpcf computed in this second way gives 
    exactly the same results as if we did not pre-compute RR.

    """
    sample1 = np.random.random((1000,3))
    sample2 = sample1
    randoms = np.random.random((100,3))
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.3,5)
    rmax = rbins.max()

    approx_cell1_size = [rmax, rmax, rmax]
    approx_cell2_size = approx_cell1_size
    approx_cellran_size = [rmax, rmax, rmax]

    normal_result = tpcf(
        sample1, rbins, sample2 = sample2, 
        randoms=randoms, period = period, 
        max_sample_size=int(1e4), estimator='Landy-Szalay', 
        approx_cell1_size=approx_cell1_size, 
        approx_cellran_size=approx_cellran_size)


    # The following quantities are computed inside the 
    # tpcf namespace. We reproduce them here because they are 
    # necessary inputs to the _random_counts and _pair_counts 
    # functions called by tpcf 
    _sample1_is_sample2 = True
    PBCs = True
    num_threads = cpu_count()
    do_DD, do_DR, do_RR = True, True, True  
    do_auto, do_cross = True, False      

    from ..tpcf import _random_counts, _pair_counts

    #count data pairs
    D1D1,D1D2,D2D2 = _pair_counts(
        sample1, sample2, rbins, period,
        num_threads, do_auto, do_cross, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size)

    #count random pairs
    D1R, D2R, RR = _random_counts(
        sample1, sample2, randoms, rbins, period,
        PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size, approx_cellran_size)

    ND1 = len(sample1)
    ND2 = len(sample2)
    NR1 = len(randoms)
    NR2 = len(randoms)


    factor1 = ND1*ND2/(NR1*NR2)
    factor2 = ND1*NR2/(NR1*NR2)

    mult = lambda x,y: x*y
    xi_11 = mult(1.0/factor1,D1D1/RR) - mult(1.0/factor2,2.0*D1R/RR) + 1.0

    # # The following assertion implies that the RR 
    # # computed within this testing namespace is the same RR 
    # # as computed in the tpcf namespace
    assert np.all(xi_11 == normal_result)

    # Now we will pass in the above RR as an argument 
    # and verify that we get an identical tpcf 
    result_with_RR_precomputed = tpcf(
        sample1, rbins, sample2 = sample2, 
        randoms=randoms, period = period, 
        max_sample_size=int(1e4), estimator='Landy-Szalay', 
        approx_cell1_size=approx_cell1_size, 
        approx_cellran_size=approx_cellran_size, 
        RR_precomputed = RR, 
        NR_precomputed = NR1)

    assert np.all(result_with_RR_precomputed == normal_result)























