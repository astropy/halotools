# cython: language_level=2
"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport ceil

from .velocity_marking_functions cimport *

__author__ = ('Andrew Hearin', 'Duncan Campbell')
__all__ = ('velocity_marked_npairs_3d_engine', )

ctypedef void (*f_type)(cnp.float64_t* w1, cnp.float64_t* w2, cnp.float64_t* shift, cnp.float64_t* result1, cnp.float64_t* result2, cnp.float64_t* result3)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def velocity_marked_npairs_3d_engine(double_mesh, x1in, y1in, z1in, x2in, y2in, z2in,
    weights1in, weights2in, int weight_func_id, rbins, cell1_tuple):
    """ Cython engine for counting pairs of points as a function of three-dimensional separation.

    Parameters
    ------------
    double_mesh : object
        Instance of `~halotools.mock_observables.RectangularDoubleMesh`

    x1in, y1in, z1in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 1

    x2in, y2in, z2in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 2

    weight_func_id : int
        weighting function integer ID.

    weights1in : array

    weights2in : array

    rbins : array
        Boundaries defining the bins in which pairs are counted.

    cell1_tuple : tuple
        Two-element tuple defining the first and last cells in
        double_mesh.mesh1 that will be looped over. Intended for use with
        python multiprocessing.

    Returns
    --------
    counts : array
        Integer array of length len(rbins) giving the number of pairs
        separated by a distance less than the corresponding entry of ``rbins``.

    """
    cdef f_type wfunc
    wfunc = return_velocity_weighting_function(weight_func_id)

    cdef cnp.float64_t[:] rbins_squared = rbins*rbins
    cdef cnp.float64_t xperiod = double_mesh.xperiod
    cdef cnp.float64_t yperiod = double_mesh.yperiod
    cdef cnp.float64_t zperiod = double_mesh.zperiod
    cdef cnp.int64_t first_cell1_element = cell1_tuple[0]
    cdef cnp.int64_t last_cell1_element = cell1_tuple[1]
    cdef int PBCs = double_mesh._PBCs

    cdef int Ncell1 = double_mesh.mesh1.ncells
    cdef int num_rbins = len(rbins)

    cdef cnp.float64_t[:] counts1 = np.zeros(num_rbins, dtype=np.float64)
    cdef cnp.float64_t[:] counts2 = np.zeros(num_rbins, dtype=np.float64)
    cdef cnp.float64_t[:] counts3 = np.zeros(num_rbins, dtype=np.float64)

    cdef cnp.float64_t[:] x1 = np.ascontiguousarray(x1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y1 = np.ascontiguousarray(y1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z1 = np.ascontiguousarray(z1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] x2 = np.ascontiguousarray(x2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y2 = np.ascontiguousarray(y2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z2 = np.ascontiguousarray(z2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:, :] weights1 = np.ascontiguousarray(weights1in[double_mesh.mesh1.idx_sorted,:], dtype=np.float64)
    cdef cnp.float64_t[:, :] weights2 = np.ascontiguousarray(weights2in[double_mesh.mesh2.idx_sorted,:], dtype=np.float64)

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

    cdef cnp.float64_t x2shift, y2shift, z2shift, dx, dy, dz, dsq, weight
    cdef cnp.float64_t x1tmp, y1tmp, z1tmp
    cdef cnp.float64_t holder1 = 0.
    cdef cnp.float64_t holder2 = 0.
    cdef cnp.float64_t holder3 = 0.
    cdef int Ni, Nj, i, j, k, l

    cdef cnp.float64_t[:] x_icell1, x_icell2
    cdef cnp.float64_t[:] y_icell1, y_icell2
    cdef cnp.float64_t[:] z_icell1, z_icell2
    cdef cnp.float64_t[:,:] w_icell1, w_icell2
    cdef cnp.float64_t[:] shift = np.zeros(3, dtype=np.float64)

    for icell1 in range(first_cell1_element, last_cell1_element):

        ifirst1 = cell1_indices[icell1]
        ilast1 = cell1_indices[icell1+1]

        #extract the points in cell1
        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]
        z_icell1 = z1[ifirst1:ilast1]

        #extract the weights in cell1
        w_icell1 = weights1[ifirst1:ilast1,:]

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
                shift[0] = x2shift

                for nonPBC_iy2 in range(leftmost_iy2, rightmost_iy2):
                    if nonPBC_iy2 < 0:
                        y2shift = -yperiod*PBCs
                    elif nonPBC_iy2 >= num_y2divs:
                        y2shift = +yperiod*PBCs
                    else:
                        y2shift = 0.
                    # Now apply the PBCs
                    iy2 = nonPBC_iy2 % num_y2divs
                    shift[1] = y2shift

                    for nonPBC_iz2 in range(leftmost_iz2, rightmost_iz2):
                        if nonPBC_iz2 < 0:
                            z2shift = -zperiod*PBCs
                        elif nonPBC_iz2 >= num_z2divs:
                            z2shift = +zperiod*PBCs
                        else:
                            z2shift = 0.
                        # Now apply the PBCs
                        iz2 = nonPBC_iz2 % num_z2divs
                        shift[2] = z2shift

                        icell2 = ix2*(num_y2divs*num_z2divs) + iy2*num_z2divs + iz2
                        ifirst2 = cell2_indices[icell2]
                        ilast2 = cell2_indices[icell2+1]

                        #extract the points in cell2
                        x_icell2 = x2[ifirst2:ilast2]
                        y_icell2 = y2[ifirst2:ilast2]
                        z_icell2 = z2[ifirst2:ilast2]

                        #extract the weights in cell2
                        w_icell2 = weights2[ifirst2:ilast2,:]

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

                                    wfunc(&w_icell1[i,0], &w_icell2[j,0], &shift[0], &holder1, &holder2, &holder3)
                                    k = num_rbins-1
                                    while dsq <= rbins_squared[k]:
                                        counts1[k] += holder1
                                        counts2[k] += holder2
                                        counts3[k] += holder3
                                        k=k-1
                                        if k<0: break

    return np.array(counts1), np.array(counts2), np.array(counts3)


cdef f_type return_velocity_weighting_function(weight_func_id):
    """
    returns a pointer to the user-specified pairwise velocity weighting function.
    """

    if weight_func_id==1:
        return relative_radial_velocity_weights
    if weight_func_id==2:
        return radial_velocity_variance_counter_weights
    if weight_func_id==3:
        return relative_los_velocity_weights
    if weight_func_id==4:
        return los_velocity_variance_counter_weights
    else:
        raise ValueError('weighting function does not exist')
