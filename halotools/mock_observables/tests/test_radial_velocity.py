"""
"""
from __future__ import absolute_import, division, print_function
import numpy as np

from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ..radial_velocity import _signed_dx, radial_distance, radial_distance_and_velocity
from ...empirical_models import enforce_periodicity_of_box

fixed_seed = 43

__all__ = ('test_signed_dx0', )


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


def test_radial_distance1():
    xs, ys, zs = 2, 2, 2
    xc, yc, zc = 1, 1, 1

    drad = radial_distance(xs, ys, zs, xc, yc, zc, np.inf)
    assert drad == np.sqrt(3)
    drad = radial_distance(xs, ys, zs, xc, yc, zc, 10)
    assert drad == np.sqrt(3)


def test_radial_distance2():
    xs, ys, zs = 9.5, 9.5, 9.5
    xc, yc, zc = 0.5, 0.5, 0.5

    drad = radial_distance(xs, ys, zs, xc, yc, zc, np.inf)
    assert drad == np.sqrt(3*81)
    drad = radial_distance(xs, ys, zs, xc, yc, zc, 10)
    assert drad == np.sqrt(3)


def test_radial_distance3():
    npts = int(1e4)
    Lbox = 150
    with NumpyRNGContext(fixed_seed):
        xc = np.random.uniform(0, Lbox, npts)
        yc = np.random.uniform(0, Lbox, npts)
        zc = np.random.uniform(0, Lbox, npts)
    xs = enforce_periodicity_of_box(xc + 1., Lbox)
    ys = enforce_periodicity_of_box(yc + 1., Lbox)
    zs = enforce_periodicity_of_box(zc + 1., Lbox)
    drad = radial_distance(xs, ys, zs, xc, yc, zc, Lbox)
    assert np.allclose(drad, np.sqrt(3))



