#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ...mock_observables import delta_sigma
from .cf_helpers import generate_locus_of_3d_points 

__all__ = ['test_delta_sigma1']

def test_delta_sigma1():
	""" Simple unit-test of delta_sigma. Does not verify correctness. 
	"""
	sample1 = np.random.random((10000, 3))
	sample2 = np.random.random((10000, 3))
	rp_bins = np.logspace(-2, -1, 5)
	pi_max = 0.1
	ds = delta_sigma(sample1, sample2, rp_bins, pi_max, period=1, log_bins=False)
