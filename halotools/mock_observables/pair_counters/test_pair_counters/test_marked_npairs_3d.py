#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pytest

from ..pairs import wnpairs as pure_python_weighted_pairs
from ..marked_npairs_3d import marked_npairs_3d, _func_signature_int_from_wfunc

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_marked_npairs_3d_periodic', )

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

def test_marked_npairs_3d_periodic():
    """
    Function tests marked_npairs with periodic boundary conditions.
    """

    rbins = np.array([0.0,0.1,0.2,0.3])

    result = marked_npairs_3d(random_sample, random_sample,
        rbins, period=period, weights1=ran_weights1, weights2=ran_weights1, weight_func_id=1)

    test_result = pure_python_weighted_pairs(random_sample, random_sample, rbins,
        period=period, weights1=ran_weights1, weights2=ran_weights1)

    assert np.allclose(test_result,result,rtol=1e-09), "pair counts are incorrect"

def test_marked_npairs_parallelization():
    """
    Function tests marked_npairs_3d with periodic boundary conditions.
    """

    rbins = np.array([0.0,0.1,0.2,0.3])

    serial_result = marked_npairs_3d(random_sample, random_sample,
        rbins, period=period, weights1=ran_weights1, weights2=ran_weights1, weight_func_id=1)

    parallel_result = marked_npairs_3d(random_sample, random_sample,
        rbins, period=period, weights1=ran_weights1, weights2=ran_weights1, weight_func_id=1,
        num_threads = 3)

    assert np.allclose(serial_result,parallel_result,rtol=1e-09), "pair counts are incorrect"











