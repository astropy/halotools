# cython: language_level=2
"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport ceil, log

from ....utils import unsorting_indices

__author__ = ('Andrew Hearin', 'Johannes U. Lange')
__all__ = ('mean_ds_12h_halo_id_engine', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mean_ds_12h_halo_id_engine(double_mesh, x1in, y1in, id1in, x2in, y2in, m2in, id2in, rp_bins, cell1_tuple):
    """ Cython engine for calculating the average excess surface density around
    points in sample 1 caused by points in sample 2.

    Parameters
    ------------
    double_mesh : object
        Instance of `~halotools.mock_observables.RectangularDoubleMesh2D`

    x1in, y1in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 1

    x2in, y2in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 2

    m2in : array
        Numpy array storing the masses of points in sample 2

    rp_bins : array_like
        numpy array of boundaries defining the bins of separation in the xy-plane
        :math:`r_{\\rm p}` in which pairs are counted.

    cell1_tuple : tuple
        Two-element tuple defining the first and last cells in
        double_mesh.mesh1 that will be looped over. Intended for use with
        python multiprocessing.

    Returns
    --------
    delta_sigma_1h : array

    delta_sigma_2h : array

    """
    cdef cnp.float64_t[:] rp_bins_squared = rp_bins*rp_bins
    cdef cnp.float64_t[:] d_log_rp_bins = np.log(rp_bins[1:] / rp_bins[:len(rp_bins)-1])
    cdef cnp.float64_t xperiod = double_mesh.xperiod
    cdef cnp.float64_t yperiod = double_mesh.yperiod
    cdef cnp.int64_t first_cell1_element = cell1_tuple[0]
    cdef cnp.int64_t last_cell1_element = cell1_tuple[1]
    cdef int PBCs = double_mesh._PBCs

    cdef int Ncell1 = double_mesh.mesh1.ncells
    cdef int num_rp_bins = len(rp_bins) - 1

    cdef cnp.float64_t[:] x1_sorted = np.ascontiguousarray(
        x1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y1_sorted = np.ascontiguousarray(
        y1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.int64_t[:] id1_sorted = np.ascontiguousarray(
        id1in[double_mesh.mesh1.idx_sorted], dtype=np.int64)

    cdef cnp.float64_t[:] x2_sorted = np.ascontiguousarray(
        x2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y2_sorted = np.ascontiguousarray(
        y2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] m2_sorted = np.ascontiguousarray(
        m2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.int64_t[:] id2_sorted = np.ascontiguousarray(
        id2in[double_mesh.mesh2.idx_sorted], dtype=np.int64)

    cdef cnp.float64_t[:, :] delta_sigma_1h = np.zeros((len(x1_sorted), num_rp_bins), dtype=np.float64)
    cdef cnp.float64_t[:, :] delta_sigma_2h = np.zeros((len(x1_sorted), num_rp_bins), dtype=np.float64)

    cdef cnp.int64_t icell1, icell2
    cdef cnp.int64_t[:] cell1_indices = np.ascontiguousarray(double_mesh.mesh1.cell_id_indices, dtype=np.int64)
    cdef cnp.int64_t[:] cell2_indices = np.ascontiguousarray(double_mesh.mesh2.cell_id_indices, dtype=np.int64)

    cdef cnp.int64_t ifirst1, ilast1, ifirst2, ilast2

    cdef int ix2, iy2, ix1, iy1
    cdef int nonPBC_ix2, nonPBC_iy2

    cdef int num_x2_covering_steps = int(np.ceil(
        double_mesh.search_xlength / double_mesh.mesh2.xcell_size))
    cdef int num_y2_covering_steps = int(np.ceil(
        double_mesh.search_ylength / double_mesh.mesh2.ycell_size))

    cdef int leftmost_ix2, rightmost_ix2
    cdef int leftmost_iy2, rightmost_iy2

    cdef int num_x1divs = double_mesh.mesh1.num_xdivs
    cdef int num_y1divs = double_mesh.mesh1.num_ydivs
    cdef int num_x2divs = double_mesh.mesh2.num_xdivs
    cdef int num_y2divs = double_mesh.mesh2.num_ydivs
    cdef int num_x2_per_x1 = num_x2divs // num_x1divs
    cdef int num_y2_per_y1 = num_y2divs // num_y1divs

    cdef cnp.float64_t x2shift, y2shift, dx, dy, dxy_sq
    cdef cnp.float64_t x1tmp, y1tmp, m2tmp
    cdef int Ni, Nj, i, j, k, l

    cdef cnp.float64_t[:] x_icell1, x_icell2
    cdef cnp.float64_t[:] y_icell1, y_icell2
    cdef cnp.float64_t[:] w_icell2

    cdef cnp.int64_t[:] id_icell1, id_icell2
    cdef cnp.int64_t id1tmp, id2tmp

    for icell1 in range(first_cell1_element, last_cell1_element):
        ifirst1 = cell1_indices[icell1]
        ilast1 = cell1_indices[icell1+1]
        x_icell1 = x1_sorted[ifirst1:ilast1]
        y_icell1 = y1_sorted[ifirst1:ilast1]
        id_icell1 = id1_sorted[ifirst1:ilast1]

        Ni = ilast1 - ifirst1
        if Ni > 0:

            ix1 = icell1 // num_y1divs
            iy1 = icell1 - ix1*num_y1divs

            leftmost_ix2 = ix1*num_x2_per_x1 - num_x2_covering_steps
            leftmost_iy2 = iy1*num_y2_per_y1 - num_y2_covering_steps

            rightmost_ix2 = (ix1+1)*num_x2_per_x1 + num_x2_covering_steps
            rightmost_iy2 = (iy1+1)*num_y2_per_y1 + num_y2_covering_steps

            for nonPBC_ix2 in range(leftmost_ix2, rightmost_ix2):
                if nonPBC_ix2 < 0:
                    x2shift = -xperiod*PBCs
                elif nonPBC_ix2 >= num_x2divs:
                    x2shift = +xperiod*PBCs
                else:
                    x2shift = 0.
                # Now apply the PBCs
                ix2 = nonPBC_ix2 % num_x2divs

                for nonPBC_iy2 in range(leftmost_iy2, rightmost_iy2):
                    if nonPBC_iy2 < 0:
                        y2shift = -yperiod*PBCs
                    elif nonPBC_iy2 >= num_y2divs:
                        y2shift = +yperiod*PBCs
                    else:
                        y2shift = 0.
                    # Now apply the PBCs
                    iy2 = nonPBC_iy2 % num_y2divs

                    icell2 = ix2*num_y2divs + iy2
                    ifirst2 = cell2_indices[icell2]
                    ilast2 = cell2_indices[icell2+1]

                    x_icell2 = x2_sorted[ifirst2:ilast2]
                    y_icell2 = y2_sorted[ifirst2:ilast2]
                    w_icell2 = m2_sorted[ifirst2:ilast2]
                    id_icell2 = id2_sorted[ifirst2:ilast2]

                    Nj = ilast2 - ifirst2
                    #loop over points in cell1 points
                    if Nj > 0:
                        for i in range(0, Ni):
                            x1tmp = x_icell1[i] - x2shift
                            y1tmp = y_icell1[i] - y2shift
                            id1tmp = id_icell1[i]
                            #loop over points in cell2 points
                            for j in range(0, Nj):
                                #calculate the square distance
                                dx = x1tmp - x_icell2[j]
                                dy = y1tmp - y_icell2[j]
                                dxy_sq = dx*dx + dy*dy

                                m2tmp = w_icell2[j]
                                id2tmp = id_icell2[j]

                                k = num_rp_bins - 1
                                if id1tmp == id2tmp:
                                    while dxy_sq <= rp_bins_squared[k + 1] and k >= 0:
                                        if dxy_sq > rp_bins_squared[k]:
                                          delta_sigma_1h[ifirst1 + i, k] -= m2tmp * (1 - log(rp_bins_squared[k+1] / dxy_sq))
                                        else:
                                          delta_sigma_1h[ifirst1 + i, k] += m2tmp * 2 * d_log_rp_bins[k]
                                        k -= 1
                                else:
                                    while dxy_sq <= rp_bins_squared[k + 1] and k >= 0:
                                        if dxy_sq > rp_bins_squared[k]:
                                          delta_sigma_2h[ifirst1 + i, k] -= m2tmp * (1 - log(rp_bins_squared[k+1] / dxy_sq))
                                        else:
                                          delta_sigma_2h[ifirst1 + i, k] += m2tmp * 2 * d_log_rp_bins[k]
                                        k -= 1


    for k in range(num_rp_bins):
        for i in range(len(x1_sorted)):
            delta_sigma_1h[i, k] /= np.pi * (rp_bins_squared[k+1] - rp_bins_squared[k])
            delta_sigma_2h[i, k] /= np.pi * (rp_bins_squared[k+1] - rp_bins_squared[k])

    # At this point, we have calculated our counts on the input arrays *after* sorting
    # Since the order of counts matters in this calculation, we need to undo the sorting
    idx_unsorted = unsorting_indices(double_mesh.mesh1.idx_sorted)
    return np.array(delta_sigma_1h)[idx_unsorted, :], np.array(delta_sigma_2h)[idx_unsorted, :]
