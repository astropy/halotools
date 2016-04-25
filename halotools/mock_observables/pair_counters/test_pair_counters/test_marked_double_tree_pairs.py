#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pytest

from copy import copy

from astropy.extern.six.moves import xrange as range

from ..pairs import wnpairs as pure_python_weighted_pairs
from ..pairs import xy_z_wnpairs as pure_python_xy_z_weighted_pairs
from ..marked_double_tree_pairs import xy_z_marked_npairs
from ..marked_double_tree_helpers import _func_signature_int_from_wfunc

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_xy_z_marked_npairs_periodic', 'test_xy_z_marked_npairs_nonperiodic')

#set up random points to test pair counters
np.random.seed(1)
Npts = 1000
random_sample = np.random.random((Npts, 3))
period = np.array([1.0, 1.0, 1.0])
num_threads = 2

#set up a regular grid of points to test pair counters
Npts2 = 10
epsilon = 0.001
gridx = np.linspace(0, 1-epsilon, Npts2)
gridy = np.linspace(0, 1-epsilon, Npts2)
gridz = np.linspace(0, 1-epsilon, Npts2)
xx, yy, zz = np.array(np.meshgrid(gridx, gridy, gridz))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()
grid_points = np.vstack([xx, yy, zz]).T

#set up random weights
ran_weights1 = np.random.random((Npts,1))
ran_weights2 = np.random.random((Npts,2))

def test_xy_z_marked_npairs_periodic():
    """
    Function tests xy_z_marked_npairs with periodic boundary conditions.
    """

    rp_bins = np.array([0.0,0.1,0.2,0.3])
    pi_bins = np.array([0.0,0.1,0.2,0.3])

    result = xy_z_marked_npairs(random_sample, random_sample,
        rp_bins, pi_bins, period=period, weights1=ran_weights1, weights2=ran_weights1, wfunc=1)

    test_result = pure_python_xy_z_weighted_pairs(random_sample, random_sample, rp_bins, pi_bins,
        period=period, weights1=ran_weights1, weights2=ran_weights1)

    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"

def test_xy_z_marked_npairs_parallelization():
    """
    Function tests xy_z_marked_npairs with periodic boundary conditions.
    """

    rp_bins = np.array([0.0,0.1,0.2,0.3])
    pi_bins = np.array([0.0,0.1,0.2,0.3])

    serial_result = xy_z_marked_npairs(random_sample, random_sample,
        rp_bins, pi_bins, period=period, weights1=ran_weights1, weights2=ran_weights1, wfunc=1)

    parallel_result = xy_z_marked_npairs(random_sample, random_sample,
        rp_bins, pi_bins, period=period, weights1=ran_weights1, weights2=ran_weights1, wfunc=1,
        num_threads = 'max')

    assert np.allclose(serial_result,parallel_result,rtol=1e-09), "pair counts are incorrect"


def test_xy_z_marked_npairs_nonperiodic():
    """
    Function tests xy_z_marked_npairs with without periodic boundary conditions.
    """

    rp_bins = np.array([0.0,0.1,0.2,0.3])
    pi_bins = np.array([0.0,0.1,0.2,0.3])

    result = xy_z_marked_npairs(random_sample, random_sample,
        rp_bins, pi_bins, period=None,
        weights1=ran_weights1, weights2=ran_weights1, wfunc=1)

    test_result = pure_python_xy_z_weighted_pairs(random_sample, random_sample,
        rp_bins, pi_bins, period=None, weights1=ran_weights1, weights2=ran_weights1)

    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"

