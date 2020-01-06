# cython: language_level=2
"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport ceil, sqrt
from libcpp.vector cimport vector

__author__ = ('Andrew Hearin', 'Duncan Campbell')
__all__ = ('pairwise_distance_3d_engine', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_distance_3d_engine(double_mesh, x1in, y1in, z1in, x2in, y2in, z2in, rmax, cell1_tuple):
    """
    Cython engine for returning pairs of points and three-dimensional separation.

    Parameters
    ------------
    double_mesh : object
        Instance of `~halotools.mock_observables.RectangularDoubleMesh`

    x1in, y1in, z1in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 1

    x2in, y2in, z2in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 2

    rmax : array
        maximum separation distance to search for and return pairs

    cell1_tuple : tuple
        Two-element tuple defining the first and last cells in
        double_mesh.mesh1 that will be looped over. Intended for use with
        python multiprocessing.

    Returns
    --------
    distance : numpy.array
        array of pairwise separation distances

    i : numpy.array
        array of 0-indexed indices in sample 1

    j : numpy.array
        array of 0-indexed indices in sample2

    """

    cdef cnp.float64_t xperiod = double_mesh.xperiod
    cdef cnp.float64_t yperiod = double_mesh.yperiod
    cdef cnp.float64_t zperiod = double_mesh.zperiod
    cdef cnp.int64_t first_cell1_element = cell1_tuple[0]
    cdef cnp.int64_t last_cell1_element = cell1_tuple[1]
    cdef int PBCs = double_mesh._PBCs

    cdef int Ncell1 = double_mesh.mesh1.ncells

    rmax = rmax*rmax
    cdef cnp.float64_t[:] rmax_squared = np.ascontiguousarray(rmax[double_mesh.mesh1.idx_sorted], dtype=np.float64)

    cdef cnp.float64_t[:] x1 = np.ascontiguousarray(x1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y1 = np.ascontiguousarray(y1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z1 = np.ascontiguousarray(z1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] x2 = np.ascontiguousarray(x2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y2 = np.ascontiguousarray(y2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z2 = np.ascontiguousarray(z2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)

    cdef vector[cnp.int_t] i_ind
    cdef vector[cnp.int_t] j_ind
    cdef vector[cnp.float64_t] distances

    cdef cnp.int64_t icell1, icell2
    cdef cnp.int64_t[:] cell1_indices = np.ascontiguousarray(double_mesh.mesh1.cell_id_indices, dtype=np.int64)
    cdef cnp.int64_t[:] cell2_indices = np.ascontiguousarray(double_mesh.mesh2.cell_id_indices, dtype=np.int64)

    cdef cnp.int64_t ifirst1, ilast1, ifirst2, ilast2

    cdef int ix2, iy2, iz2, ix1, iy1, iz1
    cdef int nonPBC_ix2, nonPBC_iy2, nonPBC_iz2

    cdef int num_x2_covering_steps = int(np.ceil(
        double_mesh.search_xlength / double_mesh.mesh2.xcell_size))
    cdef int num_y2_covering_steps = int(np.ceil(
        double_mesh.search_ylength / double_mesh.mesh2.ycell_size))
    cdef int num_z2_covering_steps = int(np.ceil(
        double_mesh.search_zlength / double_mesh.mesh2.zcell_size))

    cdef int leftmost_ix2, rightmost_ix2
    cdef int leftmost_iy2, rightmost_iy2
    cdef int leftmost_iz2, rightmost_iz2

    cdef int num_x1divs = double_mesh.mesh1.num_xdivs
    cdef int num_y1divs = double_mesh.mesh1.num_ydivs
    cdef int num_z1divs = double_mesh.mesh1.num_zdivs
    cdef int num_x2divs = double_mesh.mesh2.num_xdivs
    cdef int num_y2divs = double_mesh.mesh2.num_ydivs
    cdef int num_z2divs = double_mesh.mesh2.num_zdivs
    cdef int num_x2_per_x1 = num_x2divs // num_x1divs
    cdef int num_y2_per_y1 = num_y2divs // num_y1divs
    cdef int num_z2_per_z1 = num_z2divs // num_z1divs

    cdef cnp.float64_t x2shift, y2shift, z2shift, dx, dy, dz, dsq
    cdef cnp.float64_t x1tmp, y1tmp, z1tmp
    cdef int Ni, Nj, i, j, k, l

    cdef cnp.float64_t[:] x_icell1, x_icell2
    cdef cnp.float64_t[:] y_icell1, y_icell2
    cdef cnp.float64_t[:] z_icell1, z_icell2

    for icell1 in range(first_cell1_element, last_cell1_element):
        ifirst1 = cell1_indices[icell1]
        ilast1 = cell1_indices[icell1+1]
        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]
        z_icell1 = z1[ifirst1:ilast1]

        Ni = ilast1 - ifirst1
        if Ni > 0:

            ix1 = icell1 // (num_y1divs*num_z1divs)
            iy1 = (icell1 - ix1*num_y1divs*num_z1divs) // num_z1divs
            iz1 = icell1 - (ix1*num_y1divs*num_z1divs) - (iy1*num_z1divs)

            leftmost_ix2 = ix1*num_x2_per_x1 - num_x2_covering_steps
            leftmost_iy2 = iy1*num_y2_per_y1 - num_y2_covering_steps
            leftmost_iz2 = iz1*num_z2_per_z1 - num_z2_covering_steps

            rightmost_ix2 = (ix1+1)*num_x2_per_x1 + num_x2_covering_steps
            rightmost_iy2 = (iy1+1)*num_y2_per_y1 + num_y2_covering_steps
            rightmost_iz2 = (iz1+1)*num_z2_per_z1 + num_z2_covering_steps

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

                    for nonPBC_iz2 in range(leftmost_iz2, rightmost_iz2):
                        if nonPBC_iz2 < 0:
                            z2shift = -zperiod*PBCs
                        elif nonPBC_iz2 >= num_z2divs:
                            z2shift = +zperiod*PBCs
                        else:
                            z2shift = 0.
                        # Now apply the PBCs
                        iz2 = nonPBC_iz2 % num_z2divs

                        icell2 = ix2*(num_y2divs*num_z2divs) + iy2*num_z2divs + iz2
                        ifirst2 = cell2_indices[icell2]
                        ilast2 = cell2_indices[icell2+1]

                        x_icell2 = x2[ifirst2:ilast2]
                        y_icell2 = y2[ifirst2:ilast2]
                        z_icell2 = z2[ifirst2:ilast2]

                        Nj = ilast2 - ifirst2
                        #loop over points in cell1 points
                        if Nj > 0:
                            for i in range(0,Ni):
                                x1tmp = x_icell1[i] - x2shift
                                y1tmp = y_icell1[i] - y2shift
                                z1tmp = z_icell1[i] - z2shift
                                #loop over points in cell2 points
                                for j in range(0,Nj):
                                    #calculate the square distance
                                    dx = x1tmp - x_icell2[j]
                                    dy = y1tmp - y_icell2[j]
                                    dz = z1tmp - z_icell2[j]
                                    dsq = dx*dx + dy*dy + dz*dz

                                    if dsq <= rmax_squared[ifirst1+i]:
                                        distances.push_back(dsq)
                                        i_ind.push_back(ifirst1 + i)
                                        j_ind.push_back(ifirst2 + j)

    #input points were sorted.  return the indices of the unsorted arrays
    i_ind = np.array(i_ind).astype(int)
    i_ind = double_mesh.mesh1.idx_sorted[i_ind]
    j_ind = np.array(j_ind).astype(int)
    j_ind = double_mesh.mesh2.idx_sorted[j_ind]

    return (np.sqrt(distances).astype(float), i_ind, j_ind)
