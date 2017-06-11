""" Module providing testing for the `~halotools.mock_observables.spherical_isolation` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from ..isolation_functions_helpers import (_get_r_max,
    _set_isolation_approx_cell_sizes, _func_signature_int_from_cond_func)

from ....custom_exceptions import HalotoolsError

__all__ = ('test_get_r_max1', )

fixed_seed = 43


def test_get_r_max1():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))

    with pytest.raises(ValueError) as err:
        __ = _get_r_max(sample1, [0.1, 0.1])
    substr = "Input ``r_max`` must be the same length as ``sample1``."
    assert substr in err.value.args[0]


def test_get_r_max2():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))

    with pytest.raises(ValueError) as err:
        __ = _get_r_max(sample1, np.inf)
    substr = "Input ``r_max`` must be an array of bounded positive numbers."
    assert substr in err.value.args[0]


def test_set_isolation_approx_cell_sizes1():
    xs, ys, zs = 0.2, 0.2, 0.2
    approx_cell1_size, approx_cell2_size = 0.2, 0.2
    result = _set_isolation_approx_cell_sizes(
        approx_cell1_size, approx_cell2_size, xs, ys, zs)


def test_set_isolation_approx_cell_sizes2():
    xs, ys, zs = 0.2, 0.2, 0.2
    approx_cell1_size, approx_cell2_size = [0.2, 0.2], 0.2
    with pytest.raises(ValueError) as err:
        result = _set_isolation_approx_cell_sizes(
            approx_cell1_size, approx_cell2_size, xs, ys, zs)
    substr = "Input ``approx_cell1_size`` must be a scalar or length-3 sequence."
    assert substr in err.value.args[0]


def test_set_isolation_approx_cell_sizes3():
    xs, ys, zs = 0.2, 0.2, 0.2
    approx_cell1_size, approx_cell2_size = 0.2, [0.2, 0.2]
    with pytest.raises(ValueError) as err:
        result = _set_isolation_approx_cell_sizes(
            approx_cell1_size, approx_cell2_size, xs, ys, zs)
    substr = "Input ``approx_cell2_size`` must be a scalar or length-3 sequence."
    assert substr in err.value.args[0]


def test_func_signature_int_from_cond_func1():
    result = _func_signature_int_from_cond_func(1)

    with pytest.raises(ValueError) as err:
        __ = _func_signature_int_from_cond_func('a')
    substr = "must be one of the integer values"
    assert substr in err.value.args[0]

    with pytest.raises(HalotoolsError) as err:
        __ = _func_signature_int_from_cond_func(-1)
    substr = "must be one of the integer values"
    assert substr in err.value.args[0]
