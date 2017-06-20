"""
"""
from __future__ import absolute_import, division, print_function
import numpy as np

from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ..radial_velocity import _signed_dx, radial_distance, radial_distance_and_velocity

fixed_seed = 43


def test_signed_dx0():
    npts = 10
    xs = np.zeros(npts) + 1
    xc = np.zeros(npts) + 9
    assert np.all(_signed_dx(xs, xc, np.inf) == -8)
    assert np.all(_signed_dx(xs, xc, 10) == 2)
    assert np.all(_signed_dx(xs, 9, np.inf) == -8)
    assert np.all(_signed_dx(1, xc, np.inf) == -8)


def test_signed_dx1():
    dx = _signed_dx(2, 1, np.inf)
    assert dx == 1
    dx = _signed_dx(2, 1, 10)
    assert dx == 1


def test_signed_dx2():
    dx = _signed_dx(1, 2, np.inf)
    assert dx == -1
    dx = _signed_dx(1, 2, 10)
    assert dx == -1


def test_signed_dx3():
    dx = _signed_dx(9, 1, 10)
    assert dx == -2
    dx = _signed_dx(1, 9, 10)
    assert dx == 2
