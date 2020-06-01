# cython: language_level=2
"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport ceil
from libc.math cimport sqrt as c_sqrt


__author__ = ('Andrew Hearin', )
__all__ = ('mean_radial_velocity_vs_r_engine', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mean_radial_velocity_vs_r_engine(double_mesh, x1in, y1in, z1in, x2in, y2in, z2in,
    vx1in, vy1in, vz1in, vx2in, vy2in, vz2in,
    squared_normalize_rbins_by_in, rbins_normalized, cell1_tuple):
    """
    """
    cdef cnp.float64_t[:] rbins_normalized_squared = rbins_normalized*rbins_normalized
    cdef cnp.float64_t xperiod = double_mesh.xperiod
    cdef cnp.float64_t yperiod = double_mesh.yperiod
    cdef cnp.float64_t zperiod = double_mesh.zperiod
    cdef cnp.int64_t first_cell1_element = cell1_tuple[0]
    cdef cnp.int64_t last_cell1_element = cell1_tuple[1]
    cdef int PBCs = double_mesh._PBCs

    cdef int Ncell1 = double_mesh.mesh1.ncells
    cdef int num_rbins_normalized = len(rbins_normalized)
    cdef cnp.float64_t[:] counts = np.zeros(num_rbins_normalized, dtype=np.float64)
    cdef cnp.float64_t[:] vrad_sum = np.zeros(num_rbins_normalized, dtype=np.float64)

    cdef cnp.float64_t[:] x1 = np.ascontiguousarray(x1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y1 = np.ascontiguousarray(y1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z1 = np.ascontiguousarray(z1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] x2 = np.ascontiguousarray(x2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y2 = np.ascontiguousarray(y2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z2 = np.ascontiguousarray(z2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] vx1 = np.ascontiguousarray(vx1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] vy1 = np.ascontiguousarray(vy1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] vz1 = np.ascontiguousarray(vz1in[double_mesh.mesh1.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] vx2 = np.ascontiguousarray(vx2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] vy2 = np.ascontiguousarray(vy2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] vz2 = np.ascontiguousarray(vz2in[double_mesh.mesh2.idx_sorted], dtype=np.float64)

    cdef cnp.float64_t[:] squared_normalize_rbins_by = np.ascontiguousarray(
        squared_normalize_rbins_by_in[double_mesh.mesh1.idx_sorted], dtype=np.float64)

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

    cdef cnp.float64_t x2shift, y2shift, z2shift, dx, dy, dz, dvx, dvy, dvz, drsq, normed_drsq, vrad
    cdef cnp.float64_t x1tmp, y1tmp, z1tmp, vx1tmp, vy1tmp, vz1tmp, distance_norm1tmp
    cdef int Ni, Nj, i, j, k, l

    cdef cnp.float64_t[:] x_icell1, x_icell2
    cdef cnp.float64_t[:] y_icell1, y_icell2
    cdef cnp.float64_t[:] z_icell1, z_icell2
    cdef cnp.float64_t[:] vx_icell1, vx_icell2
    cdef cnp.float64_t[:] vy_icell1, vy_icell2
    cdef cnp.float64_t[:] vz_icell1, vz_icell2

    for icell1 in range(first_cell1_element, last_cell1_element):

        ifirst1 = cell1_indices[icell1]
        ilast1 = cell1_indices[icell1+1]

        #extract the points in cell1
        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]
        z_icell1 = z1[ifirst1:ilast1]
        vx_icell1 = vx1[ifirst1:ilast1]
        vy_icell1 = vy1[ifirst1:ilast1]
        vz_icell1 = vz1[ifirst1:ilast1]

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
                        #  Now apply the PBCs
                        iz2 = nonPBC_iz2 % num_z2divs

                        icell2 = ix2*(num_y2divs*num_z2divs) + iy2*num_z2divs + iz2
                        ifirst2 = cell2_indices[icell2]
                        ilast2 = cell2_indices[icell2+1]

                        #  extract the points in cell2
                        x_icell2 = x2[ifirst2:ilast2]
                        y_icell2 = y2[ifirst2:ilast2]
                        z_icell2 = z2[ifirst2:ilast2]
                        vx_icell2 = vx2[ifirst2:ilast2]
                        vy_icell2 = vy2[ifirst2:ilast2]
                        vz_icell2 = vz2[ifirst2:ilast2]

                        #  loop over points in cell1 points
                        Nj = ilast2 - ifirst2
                        if Nj > 0:
                            for i in range(0,Ni):
                                x1tmp = x_icell1[i] - x2shift
                                y1tmp = y_icell1[i] - y2shift
                                z1tmp = z_icell1[i] - z2shift
                                distance_norm1tmp = squared_normalize_rbins_by[i]

                                vx1tmp = vx_icell1[i]
                                vy1tmp = vy_icell1[i]
                                vz1tmp = vz_icell1[i]
                                #loop over points in cell2 points
                                for j in range(0,Nj):

                                    #  Calculate radial vector
                                    #  Note that due to the application of the shift above,
                                    #  dx gets a sign flip when PBCs are applied
                                    dx = x1tmp - x_icell2[j]
                                    dy = y1tmp - y_icell2[j]
                                    dz = z1tmp - z_icell2[j]
                                    dvx = vx1tmp - vx_icell2[j]
                                    dvy = vy1tmp - vy_icell2[j]
                                    dvz = vz1tmp - vz_icell2[j]

                                    drsq = dx*dx + dy*dy + dz*dz
                                    if drsq == 0:
                                        pass
                                    else:
                                        normed_drsq = drsq/distance_norm1tmp

                                        k = num_rbins_normalized-1
                                        if normed_drsq <= rbins_normalized_squared[k]:
                                            vrad = (dx*dvx + dy*dvy + dz*dvz)/c_sqrt(drsq)
                                        while normed_drsq <= rbins_normalized_squared[k]:
                                            vrad_sum[k] += vrad
                                            counts[k] += 1
                                            k = k-1
                                            if k < 0: break

    return np.array(counts), np.array(vrad_sum)






