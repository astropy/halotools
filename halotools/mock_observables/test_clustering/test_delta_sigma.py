#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys

from ..delta_sigma import *

import pytest
slow = pytest.mark.slow

__all__=['test_delta_sigma']

#define test sample
Npts=100
gals = np.random.random((Npts,3))
ptcls = np.random.random((Npts*10,3))
randoms = np.random.random((Npts*10,3))
period = np.array([1.0,1.0,1.0])
Lbox = np.array([1.0,1.0,1.0])
rp_bins = np.linspace(0.05,0.2,5).astype(float)
pi_max = 0.2


def test_delta_sigma():
    """
    test delta_sigma
    """
    
    result = delta_sigma(gals, ptcls, rp_bins, pi_max, period=period, log_bins=True,\
                         n_bins=25, estimator='Natural', num_threads=1)
    
    assert result.ndim ==1, 'wrong number of results returned'


