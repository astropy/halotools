"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..rectangular_mesh import RectangularDoubleMesh, sample1_cell_size

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mesh_variations', )

fixed_seed = 43


def enforce_cell_size_divide_box_size(mesh):
    xmsg = ("xcell_size = %.3f, num_xdivs = %i, xperiod = %.3f" %
        (mesh.xcell_size, mesh.num_xdivs, mesh.xperiod))
    ymsg = ("ycell_size = %.3f, num_ydivs = %i, yperiod = %.3f" %
        (mesh.ycell_size, mesh.num_ydivs, mesh.yperiod))
    zmsg = ("zcell_size = %.3f, num_zdivs = %i, zperiod = %.3f" %
        (mesh.zcell_size, mesh.num_zdivs, mesh.zperiod))

    assert np.isclose([mesh.xcell_size*mesh.num_xdivs], [mesh.xperiod]), xmsg
    assert np.isclose([mesh.ycell_size*mesh.num_ydivs], [mesh.yperiod]), ymsg
    assert np.isclose([mesh.zcell_size*mesh.num_zdivs], [mesh.zperiod]), zmsg


def enforce_correct_ncells(mesh):
    assert mesh.ncells == mesh.num_xdivs*mesh.num_ydivs*mesh.num_zdivs


def enforce_cell2_fits_in_cell1(double_mesh):
    assert double_mesh.num_xcell2_per_xcell1 == (double_mesh.mesh2.num_xdivs //
        double_mesh.mesh1.num_xdivs)
    assert double_mesh.num_ycell2_per_ycell1 == (double_mesh.mesh2.num_ydivs //
        double_mesh.mesh1.num_ydivs)
    assert double_mesh.num_zcell2_per_zcell1 == (double_mesh.mesh2.num_zdivs //
        double_mesh.mesh1.num_zdivs)

    assert double_mesh.num_xcell2_per_xcell1*double_mesh.mesh2.xcell_size == (
        double_mesh.mesh1.xcell_size)
    assert double_mesh.num_ycell2_per_ycell1*double_mesh.mesh2.ycell_size == (
        double_mesh.mesh1.ycell_size)
    assert double_mesh.num_zcell2_per_zcell1*double_mesh.mesh2.zcell_size == (
        double_mesh.mesh1.zcell_size)

    assert double_mesh.mesh2.num_xdivs >= double_mesh.mesh1.num_xdivs
    assert double_mesh.mesh2.num_ydivs >= double_mesh.mesh1.num_ydivs
    assert double_mesh.mesh2.num_zdivs >= double_mesh.mesh1.num_zdivs


def enforce_search_length_is_covered(double_mesh):
    assert double_mesh.search_xlength <= double_mesh.mesh1.xcell_size
    assert double_mesh.search_ylength <= double_mesh.mesh1.ycell_size
    assert double_mesh.search_zlength <= double_mesh.mesh1.zcell_size


def enforce_reasonable_cell_id_indices(mesh, npts):
    assert np.all(mesh.cell_id_indices <= npts)
    assert np.all(np.diff(mesh.cell_id_indices) >= 0)
    assert np.sum(np.diff(mesh.cell_id_indices)) == npts
    assert len(mesh.cell_id_indices) == mesh.ncells+1


def enforce_correct_cell_id(mesh, x, y, z):
    cell_tuple_generator = itertools.product(
        range(0, mesh.num_xdivs, int(mesh.num_xdivs/3.)),
        range(0, mesh.num_ydivs, int(mesh.num_ydivs/3.)),
        range(0, mesh.num_zdivs, int(mesh.num_zdivs/3.)))
    for ix, iy, iz in cell_tuple_generator:
        xlow, xhigh = ix*mesh.xcell_size, (ix+1)*mesh.xcell_size
        ylow, yhigh = iy*mesh.ycell_size, (iy+1)*mesh.ycell_size
        zlow, zhigh = iz*mesh.zcell_size, (iz+1)*mesh.zcell_size
        icell = ix*(mesh.num_ydivs*mesh.num_zdivs) + iy*mesh.num_zdivs + iz
        ifirst, ilast = mesh.cell_id_indices[icell], mesh.cell_id_indices[icell+1]
        assert np.all(x[ifirst:ilast] >= xlow)
        assert np.all(x[ifirst:ilast] <= xhigh)
        assert np.all(y[ifirst:ilast] >= ylow)
        assert np.all(y[ifirst:ilast] <= yhigh)
        assert np.all(z[ifirst:ilast] >= zlow)
        assert np.all(z[ifirst:ilast] <= zhigh)


def test_mesh_variations():
    npts1, npts2 = 90, 200

    period_options = (1, 7)
    approx_cell_size_multiplier_options = (1/15., 1/5., 1./np.pi)
    zc_multiplier_options = (approx_cell_size_multiplier_options[0], 0.21, 0.95)
    pbc_options = (True, False)
    search_length_options = (1/9., 1/20.)

    option_generator = itertools.product(period_options,
        zc_multiplier_options,
        approx_cell_size_multiplier_options,
        approx_cell_size_multiplier_options,
        search_length_options,
        pbc_options)

    for options in option_generator:
        period = options[0]
        xperiod, yperiod, zperiod = period, period, period
        xc1, yc1, zc1 = 0.1*xperiod, 0.1*yperiod, 0.1*zperiod
        zc_multiplier = options[1]
        zc2 = zc_multiplier*period
        points1 = generate_locus_of_3d_points(npts1, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
        points2 = generate_locus_of_3d_points(npts2, xc=xc1, yc=yc1, zc=zc2, seed=fixed_seed)
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = 3*[options[2]]
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = 3*[options[3]]
        search_xlength, search_ylength, search_zlength = 3*[options[4]]
        PBCs = options[5]
        double_mesh = RectangularDoubleMesh(
            points1[:, 0], points1[:, 1], points1[:, 2],
            points2[:, 0], points2[:, 1], points2[:, 2],
            approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
            approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
            search_xlength, search_ylength, search_zlength,
            xperiod, yperiod, zperiod, PBCs=PBCs)

        enforce_cell_size_divide_box_size(double_mesh.mesh1)
        enforce_cell_size_divide_box_size(double_mesh.mesh2)
        enforce_correct_ncells(double_mesh.mesh1)
        enforce_correct_ncells(double_mesh.mesh2)
        enforce_cell2_fits_in_cell1(double_mesh)
        enforce_search_length_is_covered(double_mesh)
        enforce_reasonable_cell_id_indices(double_mesh.mesh1, npts1)
        enforce_reasonable_cell_id_indices(double_mesh.mesh2, npts2)
        enforce_correct_cell_id(double_mesh.mesh1,
            points1[:, 0][double_mesh.mesh1.idx_sorted],
            points1[:, 1][double_mesh.mesh1.idx_sorted],
            points1[:, 2][double_mesh.mesh1.idx_sorted])


def test_sample1_cell_size():
    period, search_length, approx_cell_size = 1, 0.5, 0.1
    with pytest.raises(ValueError) as err:
        _ = sample1_cell_size(period, search_length, approx_cell_size)
    substr = "Input ``search_length`` cannot exceed period/3"
    assert substr in err.value.args[0]


def test_search_length_enforcement():
    with NumpyRNGContext(fixed_seed):
        points1 = np.random.random((100, 3))
        points2 = np.random.random((100, 3))
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = 0.1, 0.1, 0.1
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = 0.1, 0.1, 0.1
    search_xlength, search_ylength, search_zlength = 0.5, 0.5, 0.5
    xperiod, yperiod, zperiod = 1, 1, 1
    PBCs = True

    with pytest.raises(ValueError) as err:
        double_mesh = RectangularDoubleMesh(
            points1[:, 0], points1[:, 1], points1[:, 2],
            points2[:, 0], points2[:, 1], points2[:, 2],
            approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
            approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
            search_xlength, search_ylength, search_zlength,
            xperiod, yperiod, zperiod, PBCs=PBCs)
    substr = "The maximum length over which you search for pairs of points"
    assert substr in err.value.args[0]
