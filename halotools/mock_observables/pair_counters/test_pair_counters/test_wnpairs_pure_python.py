"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..pairs import wnpairs as pure_python_brute_force_wnpairs_3d
from ..pairs import xy_z_wnpairs


__all__ = ('test_wnpairs_pure_python1', )

fixed_seed = 43


def test_wnpairs_pure_python1():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 2))
    rbins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_wnpairs_3d(sample1, sample2, rbins, period=None)
    substr = "sample1 and sample2 inputs do not have the same dimension"
    assert substr in err.value.args[0]


def test_wnpairs_pure_python2():
    """
    """
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 2))
        sample2 = np.random.random((npts, 2))
    rbins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_wnpairs_3d(sample1, sample2, rbins, period=[1, 1, 1])
    substr = "period should have len == dimension of points"
    assert substr in err.value.args[0]


def test_wnpairs_pure_python3():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        weights1 = np.random.rand(npts-1)
        weights2 = np.random.rand(npts)
    rbins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_wnpairs_3d(sample1, sample2, rbins,
                    weights1=weights1, weights2=weights2)
    substr = "weights1 should have same len as sample1"
    assert substr in err.value.args[0]


def test_wnpairs_pure_python4():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        weights1 = np.random.rand(npts)
        weights2 = np.random.rand(npts-1)
    rbins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_wnpairs_3d(sample1, sample2, rbins,
                    weights1=weights1, weights2=weights2)
    substr = "weights2 should have same len as sample2"
    assert substr in err.value.args[0]


def test_xy_z_wnpairs_pure_python1():
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 2))
    rp_bins = np.linspace(0.01, 0.1, 5)
    pi_bins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = xy_z_wnpairs(sample1, sample2, rp_bins, pi_bins, period=None)
    substr = "sample1 and sample2 inputs do not have the same dimension"
    assert substr in err.value.args[0]


def test_xy_z_wnpairs_pure_python2():
    """
    """
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 2))
        sample2 = np.random.random((npts, 2))
    rp_bins = np.linspace(0.01, 0.1, 5)
    pi_bins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = xy_z_wnpairs(sample1, sample2, rp_bins, pi_bins, period=[1, 1, 1])
    substr = "period should have len == dimension of points"
    assert substr in err.value.args[0]


