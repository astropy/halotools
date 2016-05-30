#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from astropy.tests.helper import pytest

from ..mesh_helpers import _set_approximate_cell_sizes, _enforce_maximum_search_length

__all__ = ('test_set_approximate_cell_sizes', )


def test_set_approximate_cell_sizes():
    approx_cell1_size, approx_cell2_size = 0.1, 0.1
    period = 1

    with pytest.raises(ValueError) as err:
        _ = _set_approximate_cell_sizes(
            approx_cell1_size, approx_cell2_size, period)
    substr = "Input ``approx_cell1_size`` must be a length-3 sequence"
    assert substr in err.value.args[0]

    approx_cell1_size, approx_cell2_size = [0.1, 0.1, 0.1], 0.1

    with pytest.raises(ValueError) as err:
        _ = _set_approximate_cell_sizes(
            approx_cell1_size, approx_cell2_size, period)
    substr = "Input ``approx_cell2_size`` must be a length-3 sequence"
    assert substr in err.value.args[0]


def test_enforce_maximum_search_length_case1():

    search_length, period = 1, None
    _enforce_maximum_search_length(search_length, period)


def test_enforce_maximum_search_length_case2():

    search_length, period = 1, 4
    _enforce_maximum_search_length(search_length, period)


def test_enforce_maximum_search_length_case3():

    search_length, period = 1, 3
    with pytest.raises(ValueError) as err:
        _enforce_maximum_search_length(search_length, period)
    substr = "algorithm requires that the search length cannot exceed period/3 in any dimension"
    assert substr in err.value.args[0]


def test_enforce_maximum_search_length_case4():

    search_length, period = 1, (3, 4)
    with pytest.raises(ValueError) as err:
        _enforce_maximum_search_length(search_length, period)
    substr = "algorithm requires that the search length cannot exceed period/3 in any dimension"
    assert substr in err.value.args[0]


def test_enforce_maximum_search_length_case5():

    search_length, period = (1, 1), (3, 4)
    with pytest.raises(ValueError) as err:
        _enforce_maximum_search_length(search_length, period)
    substr = "algorithm requires that the search length cannot exceed period/3 in any dimension"
    assert substr in err.value.args[0]


def test_enforce_maximum_search_length_case6():

    search_length, period = (1, 1), (4, 4)
    _enforce_maximum_search_length(search_length, period)


def test_enforce_maximum_search_length_case7():

    search_length, period = (1, 4, 2), (4, 100, 7)
    _enforce_maximum_search_length(search_length, period)
