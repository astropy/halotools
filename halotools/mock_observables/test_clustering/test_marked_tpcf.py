#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..marked_tpcf import marked_tpcf

import pytest
slow = pytest.mark.slow

#create toy data to test functions
N_pts = 100
sample1 = np.random.random((N_pts,3))
sample2 = np.random.random((N_pts,3))
period = np.array([1.0,1.0,1.0])
rbins = np.linspace(0,0.3,5)
rmax = rbins.max()

__all__ = ('test_marked_tpcf_auto_periodic', 
    'test_marked_tpcf_auto_nonperiodic', 
    'test_marked_tpcf_cross1', 'test_marked_tpcf_cross_consistency')

def test_marked_tpcf_auto_periodic():
    """
    test marked_tpcf auto correlation with periodic boundary conditions
    """
    
    weight_func_id = 1
    weights1 = np.random.random(N_pts)
    
    #with randoms
    result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,
        period=period, num_threads=1, weight_func_id=weight_func_id)
    
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_marked_tpcf_auto_nonperiodic():
    """
    test marked_tpcf auto correlation without periodic boundary conditions
    """
    
    weight_func_id = 1
    weights1 = np.random.random(N_pts)
    
    #with randoms
    result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,
        period=None, num_threads=1, weight_func_id=weight_func_id)
    
    assert result.ndim == 1, "More than one correlation function returned erroneously."

def test_marked_tpcf_cross1():
    """
    """
    weights1 = np.random.random(N_pts)
    weights2 = np.random.random(N_pts)
    weight_func_id = 1

    result = marked_tpcf(sample1, rbins, sample2=sample2, 
        marks1=weights1, marks2=weights2,
        period=period, num_threads='max', weight_func_id=weight_func_id)

def test_marked_tpcf_cross_consistency():
    """
    """
    weights1 = np.random.random(N_pts)
    weights2 = np.random.random(N_pts)
    weight_func_id = 1

    cross_mark1 = marked_tpcf(sample1, rbins, sample2=sample2, 
        marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id, 
        do_auto = False, normalize_by = 'number_counts')

    auto1, cross_mark2, auto2 = marked_tpcf(sample1, rbins, sample2=sample2, 
        marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id, normalize_by = 'number_counts')

    auto1b, auto2b = marked_tpcf(sample1, rbins, sample2=sample2, 
        marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id, 
        do_cross = False, normalize_by = 'number_counts')

    assert np.all(cross_mark1 == cross_mark2)



