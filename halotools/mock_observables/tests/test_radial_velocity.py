"""
"""
from __future__ import absolute_import, division, print_function
import numpy as np

from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from .cf_helpers import generate_thin_shell_of_3d_points
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


def test_radial_distance4():
    """
    """
    npts = 100
    xc, yc, zc = 9, 9, 9
    radius = 2.
    Lbox = 10.
    pts = generate_thin_shell_of_3d_points(npts, radius, xc, yc, zc)
    xs = enforce_periodicity_of_box(pts[:, 0], Lbox)
    ys = enforce_periodicity_of_box(pts[:, 1], Lbox)
    zs = enforce_periodicity_of_box(pts[:, 2], Lbox)
    drad = radial_distance(xs, ys, zs, xc, yc, zc, Lbox)
    assert np.allclose(drad, radius)
    drad = radial_distance(xs, ys, zs, xc, yc, zc, np.inf)
    assert not np.allclose(drad, radius)


def test_radial_velocity1():
    Lbox = np.inf
    xc, yc, zc = 5., 5., 5.
    vxc, vyc, vzc = 0., 0., 0.
    input_drad = 1.
    xs, ys, zs = xc + input_drad, yc, zc
    input_vrad = -1.
    vxs, vys, vzs = vxc + input_vrad, vyc, vzc
    inferred_drad, inferred_vrad = radial_distance_and_velocity(xs, ys, zs, vxs, vys, vzs,
            xc, yc, zc, vxc, vyc, vzc, Lbox)
    assert np.allclose(inferred_drad, input_drad)
    assert np.allclose(inferred_vrad, input_vrad)


def test_radial_velocity2():
    Lbox = 5
    xc, yc, zc = 4.9, 4.9, 4.9
    vxc, vyc, vzc = 0., 0., 0.
    input_drad = 1.
    xs, ys, zs = xc + input_drad - Lbox, yc, zc
    input_vrad = -1.
    vxs, vys, vzs = vxc + input_vrad, vyc, vzc
    inferred_drad, inferred_vrad = radial_distance_and_velocity(xs, ys, zs, vxs, vys, vzs,
            xc, yc, zc, vxc, vyc, vzc, Lbox)
    assert np.allclose(inferred_drad, input_drad)
    assert np.allclose(inferred_vrad, input_vrad)


def test_radial_velocity3():
    """
    """
    npts = 100
    xc, yc, zc = 9., 9., 9.
    vxc, vyc, vzc = 0, 0, 0
    Lbox = 10.
    input_drad = 3.
    xs = np.zeros(npts) + xc + input_drad
    ys = np.zeros(npts) + yc + input_drad
    zs = np.zeros(npts) + zc
    input_vrad = -1.
    vxs = np.zeros(npts) + input_vrad
    vys = np.zeros(npts) + input_vrad
    vzs = np.zeros(npts)
    xs, vxs = enforce_periodicity_of_box(xs, Lbox, velocity=vxs)
    ys, vys = enforce_periodicity_of_box(ys, Lbox, velocity=vys)
    zs, vzs = enforce_periodicity_of_box(zs, Lbox, velocity=vzs)

    inferred_drad, inferred_vrad = radial_distance_and_velocity(xs, ys, zs, vxs, vys, vzs,
            xc, yc, zc, vxc, vyc, vzc, Lbox)
    assert np.allclose(inferred_drad, input_drad*np.sqrt(2))
    correct_vrad = input_vrad*np.sqrt(2)
    assert np.allclose(correct_vrad, inferred_vrad)






