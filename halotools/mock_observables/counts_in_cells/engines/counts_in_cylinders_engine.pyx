# cython: language_level=2
"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport ceil

from .conditions.conditions cimport Condition
from .conditions.conditions import choose_condition
from ....utils import unsorting_indices

__author__ = ('Andrew Hearin', )
__all__ = ('counts_in_cylinders_engine', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def counts_in_cylinders_engine(
        double_mesh,
        x1in, y1in, z1in,
        x2in, y2in, z2in,
        rp_max, pi_max,
        return_indexes,
        condition, condition_args,
        cell1_tuple):
    """
    Cython engine for determining counting the number of points in ``sample2``
    in a cylinder surrounding each point in ``sample1``.

    Parameters
    ----------
    double_mesh : object
        Instance of `~halotools.mock_observables.RectangularDoubleMesh`

    x1in : numpy.array
        Length-Npts1 array storing Cartesian x-coordinates of points of 'sample 1'

    y1in : numpy.array
        Length-Npts1 array storing Cartesian y-coordinates of points of 'sample 1'

    z1in : numpy.array
        Length-Npts1 array storing Cartesian z-coordinates of points of 'sample 1'

    x2in : numpy.array
        Length-Npts2 array storing Cartesian x-coordinates of points of 'sample 2'

    y2in : numpy.array
        Length-Npts2 array storing Cartesian y-coordinates of points of 'sample 2'

    z2in : numpy.array
        Length-Npts2 array storing Cartesian z-coordinates of points of 'sample 2'

    rp_max : numpy.array
        Length-Npts1 array storing the x-y projected radial distance,
        i.e., the radius of cylinder, to search
        for neighbors around each point in 'sample 1'

    pi_max : numpy.array
        Length-Npts1 array storing the z distance,
        i.e., the half the length of a cylinder,
        to search for neighbors around each point in 'sample 1'

    return_indexes : bool
        If true, return both counts and the indexes of the pairs.

    condition : str, optional
        Require a condition to be met for a pair to be counted.
        See options below:
        None | "always_true":
            Count all pairs in cylinder

        "mass_frac":
            Only count pairs which satisfy lim[0] < mass2/mass1 < lim[1]

    condition_args : tuple, optional
        Arguments passed to the condition constructor
        "always_true":
            *args will be ignored

        "mass_frac":
            -mass1 (array of mass of sample 1; required)
            -mass2 (array of mass of sample 2; required)
            -lim (tuple of min,max; required)
            -lower_equality (bool to use lim[0] <= m2/m1; optional)
            -upper_equality (bool to use m2/m1 <= lim[1]; optional)

    cell1_tuple : tuple
        Two-element tuple defining the first and last cells in
        double_mesh.mesh1 that will be looped over. Intended for use with
        python multiprocessing.

    Returns
    -------
    counts : numpy.array
        Length-Npts1 integer array storing the number of ``sample2`` points
        inside a cylinder centered at each point in ``sample1``.

    indexes : numpy.array
        Num pairs length structured array with column ``i1``, the index of the
        sample 1 point, and column ``i2``, the index of the sample 2 point in
        in that cylinder
    """

    rp_max_squared_tmp = rp_max*rp_max
    cdef cnp.float64_t[:] rp_max_squared = np.ascontiguousarray(
        rp_max_squared_tmp[double_mesh.mesh1.idx_sorted])
    pi_max_squared_tmp = pi_max*pi_max
    cdef cnp.float64_t[:] pi_max_squared = np.ascontiguousarray(
        pi_max_squared_tmp[double_mesh.mesh1.idx_sorted])

    # Store the original order to use during loop
    cdef cnp.int64_t[:] idx_sorted1 = double_mesh.mesh1.idx_sorted
    cdef cnp.int64_t[:] idx_sorted2 = double_mesh.mesh2.idx_sorted
    cdef cnp.int64_t i_original
    cdef cnp.int64_t j_original

    # Choose the condition function
    cdef Condition cond = choose_condition(condition, condition_args)


    cdef cnp.float64_t xperiod = double_mesh.xperiod
    cdef cnp.float64_t yperiod = double_mesh.yperiod
    cdef cnp.float64_t zperiod = double_mesh.zperiod
    cdef cnp.int64_t first_cell1_element = cell1_tuple[0]
    cdef cnp.int64_t last_cell1_element = cell1_tuple[1]
    cdef int PBCs = double_mesh._PBCs

    cdef int Ncell1 = double_mesh.mesh1.ncells
    cdef int Npts1 = len(x1in)

    cdef cnp.float64_t[:] x1_sorted = np.ascontiguousarray(
        x1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y1_sorted = np.ascontiguousarray(
        y1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z1_sorted = np.ascontiguousarray(
        z1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] x2_sorted = np.ascontiguousarray(
        x2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y2_sorted = np.ascontiguousarray(
        y2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z2_sorted = np.ascontiguousarray(
        z2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)

    cdef bint c_return_indexes = return_indexes
    cdef cnp.int64_t[:] counts = np.zeros(len(x1_sorted), dtype=np.int64)
    cdef int current_indexes_cnt = 0
    cdef int current_indexes_len = len(x1_sorted) if c_return_indexes else 0
    cdef cnp.uint32_t[:,:] indexes = np.ascontiguousarray(
            np.zeros((current_indexes_len, 2), dtype=np.uint32))

    cdef cnp.int64_t icell1, icell2
    cdef cnp.int64_t[:] cell1_indices = np.ascontiguousarray(
        double_mesh.mesh1.cell_id_indices, dtype=np.int64)
    cdef cnp.int64_t[:] cell2_indices = np.ascontiguousarray(
        double_mesh.mesh2.cell_id_indices, dtype=np.int64)

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

    cdef cnp.float64_t x2shift, y2shift, z2shift, dx, dy, dz, dsq, dxy_sq, dz_sq
    cdef cnp.float64_t x1tmp, y1tmp, z1tmp, rp_max_squaredtmp, pi_max_squaredtmp
    cdef int Ni, Nj, i, j, k, l, current_data1_index

    cdef cnp.float64_t[:] x_icell1, x_icell2
    cdef cnp.float64_t[:] y_icell1, y_icell2
    cdef cnp.float64_t[:] z_icell1, z_icell2

    for icell1 in range(first_cell1_element, last_cell1_element):
        ifirst1 = cell1_indices[icell1]
        ilast1 = cell1_indices[icell1+1]

        x_icell1 = x1_sorted[ifirst1:ilast1]
        y_icell1 = y1_sorted[ifirst1:ilast1]
        z_icell1 = z1_sorted[ifirst1:ilast1]

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

                        x_icell2 = x2_sorted[ifirst2:ilast2]
                        y_icell2 = y2_sorted[ifirst2:ilast2]
                        z_icell2 = z2_sorted[ifirst2:ilast2]

                        Nj = ilast2 - ifirst2
                        #loop over points in cell1
                        if Nj > 0:
                            for i in range(0,Ni):
                                x1tmp = x_icell1[i] - x2shift
                                y1tmp = y_icell1[i] - y2shift
                                z1tmp = z_icell1[i] - z2shift
                                rp_max_squaredtmp = rp_max_squared[ifirst1+i]
                                pi_max_squaredtmp = pi_max_squared[ifirst1+i]

                                #loop over points in cell2
                                for j in range(0,Nj):
                                    #calculate the square distance
                                    dx = x1tmp - x_icell2[j]
                                    dy = y1tmp - y_icell2[j]
                                    dz = z1tmp - z_icell2[j]
                                    dxy_sq = dx*dx + dy*dy
                                    dz_sq = dz*dz

                                    if (dxy_sq < rp_max_squaredtmp) & (dz_sq < pi_max_squaredtmp):

                                        i_original = idx_sorted1[ifirst1+i]
                                        j_original = idx_sorted2[ifirst2+j]
                                        if cond.func(i_original, j_original):
                                            counts[ifirst1+i] += 1

                                            if c_return_indexes:
                                                indexes[current_indexes_cnt, 0] = ifirst1+i
                                                indexes[current_indexes_cnt, 1] = ifirst2+j
                                                current_indexes_cnt += 1
                                                if current_indexes_cnt == current_indexes_len:
                                                    current_indexes_len *= 2
                                                    indexes = np.resize(indexes, (current_indexes_len, 2))

    # At this point, we have calculated our pairs on the input arrays *after* sorting
    # Since the order matters in this calculation, we need to undo the sorting
    # We also need to reassign these to a non-cdef'ed variables so they can be pickled for pool
    counts_uns = np.array(counts)[unsorting_indices(double_mesh.mesh1.idx_sorted)]
    if c_return_indexes:
        # https://github.com/numpy/numpy/issues/2407 for str("colname")
        np_indexes = np.asarray(indexes[:current_indexes_cnt]).flatten().view(
	            dtype=[(str("i1"), np.uint32), (str("i2"), np.uint32)])
        np_indexes["i1"] = double_mesh.mesh1.idx_sorted[np_indexes["i1"]]
        np_indexes["i2"] = double_mesh.mesh2.idx_sorted[np_indexes["i2"]]
        return counts_uns, np_indexes

    return counts_uns
