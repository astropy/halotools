#cython: language_level=2
"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport ceil

__author__ = ('Andrew Hearin', )
__all__ = ('weighted_npairs_xy_engine', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def weighted_npairs_xy_engine(double_mesh, x1in, y1in, x2in, y2in, w2in, rp_bins, cell1_tuple):
    """ Cython engine for counting pairs of points as a function of projected separation.

    Parameters
    ------------
    double_mesh : object
        Instance of `~halotools.mock_observables.RectangularDoubleMesh2D`

    x1in, y1in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 1

    x2in, y2in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 2

    w2in : array
        Numpy array storing the weights of points in sample 2

    rp_bins : array_like
        numpy array of boundaries defining the bins of separation in the xy-plane
        :math:`r_{\\rm p}` in which pairs are counted.

    cell1_tuple : tuple
        Two-element tuple defining the first and last cells in
        double_mesh.mesh1 that will be looped over. Intended for use with
        python multiprocessing.

    Returns
    --------
    weighted_counts : array
        Array of length len(rp_bins) giving the weighted sum of pairs
        separated by a distance less than the corresponding entry of ``rp_bins``.

    """
    cdef cnp.float64_t[:] rp_bins_squared = rp_bins*rp_bins
    cdef cnp.float64_t xperiod = double_mesh.xperiod
    cdef cnp.float64_t yperiod = double_mesh.yperiod
    cdef cnp.int64_t first_cell1_element = cell1_tuple[0]
    cdef cnp.int64_t last_cell1_element = cell1_tuple[1]
    cdef int PBCs = double_mesh._PBCs

    cdef int Ncell1 = double_mesh.mesh1.ncells
    cdef int num_rp_bins = len(rp_bins)
    cdef cnp.float64_t[:] weighted_counts = np.zeros(num_rp_bins, dtype=np.float64)

    cdef cnp.float64_t[:] x1 = np.ascontiguousarray(x1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y1 = np.ascontiguousarray(y1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] x2 = np.ascontiguousarray(x2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y2 = np.ascontiguousarray(y2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] w2 = np.ascontiguousarray(w2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)

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
    cdef cnp.float64_t x1tmp, y1tmp, w2tmp
    cdef int Ni, Nj, i, j, k, l

    cdef cnp.float64_t[:] x_icell1, x_icell2
    cdef cnp.float64_t[:] y_icell1, y_icell2
    cdef cnp.float64_t[:] w_icell2

    for icell1 in range(first_cell1_element, last_cell1_element):
        ifirst1 = cell1_indices[icell1]
        ilast1 = cell1_indices[icell1+1]
        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]

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

                    x_icell2 = x2[ifirst2:ilast2]
                    y_icell2 = y2[ifirst2:ilast2]
                    w_icell2 = w2[ifirst2:ilast2]

                    Nj = ilast2 - ifirst2
                    #loop over points in cell1 points
                    if Nj > 0:
                        for i in range(0, Ni):
                            x1tmp = x_icell1[i] - x2shift
                            y1tmp = y_icell1[i] - y2shift
                            #loop over points in cell2 points
                            for j in range(0, Nj):
                                #calculate the square distance
                                dx = x1tmp - x_icell2[j]
                                dy = y1tmp - y_icell2[j]
                                dxy_sq = dx*dx + dy*dy

                                w2tmp = w_icell2[j]

                                k = num_rp_bins-1
                                while dxy_sq <= rp_bins_squared[k]:
                                    weighted_counts[k] += w2tmp
                                    k=k-1
                                    if k<0: break

    return np.array(weighted_counts)











