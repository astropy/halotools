"""
"""
from __future__ import absolute_import, division, print_function
import numpy as np

from astropy.utils.misc import NumpyRNGContext

from ...tests.cf_helpers import generate_thin_shell_of_3d_points
from ..radial_velocity_decomposition import _signed_dx, radial_distance, radial_distance_and_velocity
from ....empirical_models import enforce_periodicity_of_box

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
    """ Put a central at (5, 5, 5) in a box WITHOUT periodic boundary conditions,
    with the central moving in the direction (3, 0, 0).
    Place all satellites at the point (6, 5, 5), moving in the direction (2, 0, 0),
    i.e., in the negative radial direction that is aligned with the x-dimension.
    Verify that we recover the correct radial velocity of -1 when ignoring PBCs

    """
    Lbox = np.inf
    xc, yc, zc = 5., 5., 5.
    vxc, vyc, vzc = 3., 0., 0.
    input_drad = 1.
    xs, ys, zs = xc + input_drad, yc, zc
    vxs, vys, vzs = 2., vyc, vzc
    input_vrad = vxs - vxc
    inferred_drad, inferred_vrad = radial_distance_and_velocity(xs, ys, zs, vxs, vys, vzs,
            xc, yc, zc, vxc, vyc, vzc, Lbox)
    assert np.allclose(inferred_drad, input_drad)
    assert np.allclose(inferred_vrad, input_vrad)
    assert input_vrad == -1


def test_radial_velocity2():
    """ Put a central at (4.9, 4.9, 4.9) in a box of length 10.
    Place all satellites at the point (0.9, 4.9, 4.9), moving in the direction (-1, 0, 0),
    i.e., in the negative radial direction that is aligned with the x-dimension.
    Verify that we recover the correct radial velocity after applying PBCs

    """
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
    """ Put a central at (9, 9, 9) in a box of length 10.
    Place all satellites at the point (11, 11, 9), moving in the direction (-1, -1, 0),
    i.e., in the negative radial direction away from the (stationary) central.
    Verify that we recover the correct radial velocity after applying PBCs

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


def test_radial_velocity4():
    """ Pick a point near the edge of the box for the central,
    and surround it by a spherical shell of satellites with a radius chosen so
    that some of the satellites are scattered outside the box, giving a random
    velocity to the central and random radial velocity to each satellite.
    Verify that we recover the input radial velocity.
    """
    Lbox = 250.
    npts2 = 500
    xc, yc, zc = Lbox - 1, Lbox - 1, Lbox - 1

    radius = 2.
    pts2 = generate_thin_shell_of_3d_points(npts2, radius, xc, yc, zc, seed=fixed_seed)
    xsarr, ysarr, zsarr = pts2[:, 0], pts2[:, 1], pts2[:, 2]
    normed_dxsarr = (xsarr - xc)/radius
    normed_dysarr = (ysarr - yc)/radius
    normed_dzsarr = (zsarr - zc)/radius

    input_vrad = np.random.uniform(-100, 100, npts2)
    dvxsarr = normed_dxsarr*input_vrad
    dvysarr = normed_dysarr*input_vrad
    dvzsarr = normed_dzsarr*input_vrad
    with NumpyRNGContext(fixed_seed):
        vxc = np.random.uniform(-100, 100)
        vyc = np.random.uniform(-100, 100)
        vzc = np.random.uniform(-100, 100)
    vxsarr = vxc + dvxsarr
    vysarr = vyc + dvysarr
    vzsarr = vzc + dvzsarr

    xsarr, vxsarr = enforce_periodicity_of_box(xsarr, Lbox, velocity=vxsarr)
    ysarr, vysarr = enforce_periodicity_of_box(ysarr, Lbox, velocity=vysarr)
    zsarr, vzsarr = enforce_periodicity_of_box(zsarr, Lbox, velocity=vzsarr)

    inferred_drad, inferred_vrad = radial_distance_and_velocity(
        xsarr, ysarr, zsarr, vxsarr, vysarr, vzsarr,
            xc, yc, zc, vxc, vyc, vzc, Lbox)

    assert np.allclose(inferred_drad, radius)
    assert np.allclose(inferred_vrad, input_vrad)

