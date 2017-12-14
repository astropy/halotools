""" Module providing testing of `halotools.mock_observables.velocity_marked_npairs_3d`
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..velocity_marked_npairs_3d import velocity_marked_npairs_3d

from ..velocity_marked_npairs_3d import _velocity_marked_npairs_3d_process_weights as process_weights_3d

__all__ = ('test_velocity_marked_npairs_3d_test1', )

fixed_seed = 43


def test_velocity_marked_npairs_3d_test1():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        weights1 = np.random.random((npts, 6))

    weight_func_id = 1
    __ = process_weights_3d(sample1, sample1, weights1, weights1, weight_func_id)


def test_velocity_marked_npairs_3d_test2():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        weights1 = np.random.random((npts, 6))
        weights2 = np.random.random((npts, 6))

    weight_func_id = 1
    __ = process_weights_3d(sample1, sample2, weights1, weights2, weight_func_id)


def test_velocity_marked_npairs_3d_test3():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        weights1 = np.random.random((npts, 7))
        weights2 = np.random.random((npts, 7))

    weight_func_id = 1
    with pytest.raises(ValueError) as err:
        __ = process_weights_3d(sample1, sample2, weights1, weights2, weight_func_id)
    substr = "For this value of `weight_func_id`, there should be"
    assert substr in err.value.args[0]


def test_velocity_marked_npairs_3d_test4():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        weights1 = np.random.random(npts)
        weights2 = np.random.random(npts)

    weight_func_id = 1
    with pytest.raises(ValueError) as err:
        __ = process_weights_3d(sample1, sample2, weights1, weights2, weight_func_id)
    substr = "does not have the correct length. "
    assert substr in err.value.args[0]


def test_velocity_marked_npairs_3d_test5():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        weights1 = np.random.random((npts, 3))
        weights2 = np.random.random((npts, 3))

    weight_func_id = 1
    with pytest.raises(ValueError) as err:
        __ = process_weights_3d(sample1, sample2, weights1, weights2, weight_func_id)
    substr = "For this value of `weight_func_id`, there should be "
    assert substr in err.value.args[0]


def test_velocity_marked_npairs_3d_test6():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        weights1 = np.random.random((npts, 3, 4))
        weights2 = np.random.random((npts, 3, 4))

    weight_func_id = 1
    with pytest.raises(ValueError) as err:
        __ = process_weights_3d(sample1, sample2, weights1, weights2, weight_func_id)
    substr = "You must either pass in a 1-D or 2-D array"
    assert substr in err.value.args[0]
