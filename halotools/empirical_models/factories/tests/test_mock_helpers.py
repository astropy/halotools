"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.table import Table

from ..mock_helpers import infer_mask_from_kwargs
from ....custom_exceptions import HalotoolsError

__all__ = ('test_infer_mask_from_kwargs_consistency', )


def test_infer_mask_from_kwargs_consistency():
    x = np.arange(10)
    t = Table({'x': x})
    mask1 = infer_mask_from_kwargs(t, x=4)

    def f(t):
        return t['x'] == 4

    mask2 = infer_mask_from_kwargs(t, mask_function=f)
    assert np.all(mask1 == mask2)


def test_infer_trivial_mask_from_kwargs():
    x = np.arange(10)
    t = Table({'x': x})
    mask = infer_mask_from_kwargs(t)
    assert len(t) == len(t[mask])


def test_too_many_args():
    x = np.arange(10)
    t = Table({'x': x, 'y': x})
    with pytest.raises(HalotoolsError) as err:
        mask = infer_mask_from_kwargs(t, x=4, y=5)
    substr = "Only a single mask at a time is permitted "
    assert substr in err.value.args[0]
