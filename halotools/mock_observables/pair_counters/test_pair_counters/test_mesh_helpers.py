#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

__all__ = ('test_set_approximate_cell_sizes', )

import pytest 

from ..mesh_helpers import _set_approximate_cell_sizes

def test_set_approximate_cell_sizes():
    approx_cell1_size, approx_cell2_size = 0.1, 0.1
    rmax, period = 0.2, 1

    with pytest.raises(ValueError) as err:
        _ = _set_approximate_cell_sizes(
            approx_cell1_size, approx_cell2_size, rmax, period)
    substr = "Input ``approx_cell1_size`` must be a length-3 sequence"
    assert substr in err.value.args[0]

    approx_cell1_size, approx_cell2_size = [0.1, 0.1, 0.1], 0.1

    with pytest.raises(ValueError) as err:
        _ = _set_approximate_cell_sizes(
            approx_cell1_size, approx_cell2_size, rmax, period)
    substr = "Input ``approx_cell2_size`` must be a length-3 sequence"
    assert substr in err.value.args[0]
