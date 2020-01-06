# cython: language_level=2
"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport ceil

__author__ = ('Andrew Hearin', 'Duncan Campbell')
__all__ = ('npairs_jackknife_3d_engine', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def npairs_jackknife_3d_engine(double_mesh, x1in, y1in, z1in, x2in, y2in, z2in,
    weights1in, weights2in, jtags1in, jtags2in, cnp.int64_t N_samples, rbins, cell1_tuple):
    """ Cython engine for counting pairs of points as a function of three-dimensional separation.

    Parameters
    ------------
    double_mesh : object
        Instance of `~halotools.mock_observables.RectangularDoubleMesh`

    x1in, y1in, z1in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 1

    x2in, y2in, z2in : arrays
        Numpy arrays storing Cartesian coordinates of points in sample 2

    weights1in : array
        Numpy array storing the weights for points in sample 1

    weights2in : array
        Numpy array storing the weights for points in sample 2

    jtags1in : array
        Numpy array storing the subvolume label integers for points in sample 1

    jtags2in : array
        Numpy array storing the subvolume label integers for points in sample 2

    N_samples : int
        Total number of cells into which the simulated box has been subdivided

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
    cdef cnp.float64_t[:] rbins_squared = rbins*rbins
    cdef cnp.float64_t xperiod = double_mesh.xperiod
    cdef cnp.float64_t yperiod = double_mesh.yperiod
    cdef cnp.float64_t zperiod = double_mesh.zperiod
    cdef cnp.int64_t first_cell1_element = cell1_tuple[0]
    cdef cnp.int64_t last_cell1_element = cell1_tuple[1]
    cdef int PBCs = double_mesh._PBCs

    cdef int Ncell1 = double_mesh.mesh1.ncells
    cdef int num_rbins = len(rbins)
    cdef cnp.float64_t[:,:] counts = np.zeros((N_samples+1, num_rbins), dtype=np.float64)

    cdef cnp.float64_t[:] x1 = np.ascontiguousarray(x1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y1 = np.ascontiguousarray(y1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z1 = np.ascontiguousarray(z1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] x2 = np.ascontiguousarray(x2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y2 = np.ascontiguousarray(y2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z2 = np.ascontiguousarray(z2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)

    cdef cnp.float64_t[:] weights1 = np.ascontiguousarray(weights1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] weights2 = np.ascontiguousarray(weights2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.int64_t[:] jtags1 = np.ascontiguousarray(jtags1in[double_mesh.mesh1.idx_sorted], dtype=np.int64)
    cdef cnp.int64_t[:] jtags2 = np.ascontiguousarray(jtags2in[double_mesh.mesh2.idx_sorted], dtype=np.int64)

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

    cdef cnp.int64_t j1, j2
    cdef cnp.float64_t w1, w2

    cdef int Ni, Nj, i, j, k, l
    cdef cnp.int64_t s

    cdef cnp.float64_t[:] x_icell1, x_icell2
    cdef cnp.float64_t[:] y_icell1, y_icell2
    cdef cnp.float64_t[:] z_icell1, z_icell2
    cdef cnp.float64_t[:] w_icell1, w_icell2
    cdef cnp.int64_t[:] j_icell1, j_icell2

    for icell1 in range(first_cell1_element, last_cell1_element):
        ifirst1 = cell1_indices[icell1]
        ilast1 = cell1_indices[icell1+1]

        #extract the points in cell1
        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]
        z_icell1 = z1[ifirst1:ilast1]

        #extract the weights in cell1
        w_icell1 = weights1[ifirst1:ilast1]

        #extract the subvolume tags in cell1
        j_icell1 = jtags1[ifirst1:ilast1]

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

                        #extract the points in cell2
                        x_icell2 = x2[ifirst2:ilast2]
                        y_icell2 = y2[ifirst2:ilast2]
                        z_icell2 = z2[ifirst2:ilast2]

                        #extract the weights in cell1
                        w_icell2 = weights2[ifirst2:ilast2]

                        #extract the subvolume tags in cell1
                        j_icell2 = jtags2[ifirst2:ilast2]

                        Nj = ilast2 - ifirst2
                        #loop over points in cell1
                        if Nj > 0:
                            for i in range(0,Ni):
                                x1tmp = x_icell1[i] - x2shift
                                y1tmp = y_icell1[i] - y2shift
                                z1tmp = z_icell1[i] - z2shift

                                w1 = w_icell1[i]
                                j1 = j_icell1[i]
                                #loop over points in cell2
                                for j in range(0,Nj):
                                    #calculate the square distance
                                    dx = x1tmp - x_icell2[j]
                                    dy = y1tmp - y_icell2[j]
                                    dz = z1tmp - z_icell2[j]
                                    dsq = dx*dx + dy*dy + dz*dz

                                    w2 = w_icell2[j]
                                    j2 = j_icell2[j]

                                    for s in range(N_samples+1):
                                        k = num_rbins-1
                                        while dsq<=rbins_squared[k]:
                                            counts[s,k] += jweight(s, j1, j2, w1, w2)
                                            k=k-1
                                            if k<0: break


    return np.array(counts)


cdef inline cnp.float64_t jweight(cnp.int64_t j, cnp.int64_t j1, cnp.int64_t j2,
    cnp.float64_t w1, cnp.float64_t w2):
    """
    Return the jackknife weighted count.

    parameters
    ----------
    j : int
        subsample being removed

    j1 : int
        integer label indicating which subsample point 1 occupies

    j2 : int
        integer label indicating which subsample point 2 occupies

    w1 : float
        weight associated with point 1

    w2 : float
        weight associated with point 2

    Returns
    -------
    w : double
        0.0, w1*w2*0.5, or w1*w2

    Notes
    -----
    We use the tag '0' to indicated we want to use the entire sample, i.e. no subsample
    should be labeled with a '0'.

    jackknife wiehgt is caclulated as follows:
    if both points are inside the sample, return w1*w2
    if both points are outside the sample, return 0.0
    if one point is within and one point is outside the sample, return 0.5*w1*w2
    """
    cdef cnp.float64_t result
    if j==0:
        result = w1 * w2
    # both outside the sub-sample
    elif (j1 == j2) & (j1 == j):
        result = 0.0
    # both inside the sub-sample
    elif (j1 != j) & (j2 != j):
        result = (w1 * w2)
    # only one inside the sub-sample
    elif (j1 != j2) & ((j1 == j) | (j2 == j)):
        result = 0.5*(w1 * w2)

    return result



