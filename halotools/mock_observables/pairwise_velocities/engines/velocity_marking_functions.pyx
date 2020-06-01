# cython: language_level=2
# cython: profile=False
"""
weighting fuctions that return pairwise velocity calculations.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
cimport numpy as cnp

__all__= ("relative_radial_velocity_weights",
    "radial_velocity_variance_counter_weights",
    "relative_los_velocity_weights",
    "los_velocity_variance_counter_weights")

__author__ = ("Duncan Campbell", "Andrew Hearin")


cdef void relative_radial_velocity_weights(cnp.float64_t* w1,
                                           cnp.float64_t* w2,
                                           cnp.float64_t* shift,
                                           cnp.float64_t* result1,
                                           cnp.float64_t* result2,
                                           cnp.float64_t* result3):
    """
    Calculate the relative radial velocity between two points.

    func ID=1

    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0:3] x,y,z positions
        w1[3:6] vx, vy, vz velocities

    w2 : pointer to an array
        weights array associated with data2
        w2[0:3] x,y,z positions
        w2[3:6] vx, vy, vz velocities

    shift : pointer to an array
        Length-3 array storing the amount the points were shifted in each spatial
        dimension.  This is used when doing pair counts on periodic boxes and the
        points have been preshifted.

    result1 : pointer to a double
        relative radial velocity

    result2 : pointer to a double
        0.0 (dummy)

    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)

    """
    #calculate radial vector between points
    # Note that due to the application of the shift,
    #   when PBCs are applied, rx, ry, rz has its normal sign flipped
    cdef cnp.float64_t rx = w1[0] - (w2[0] + shift[0])
    cdef cnp.float64_t ry = w1[1] - (w2[1] + shift[1])
    cdef cnp.float64_t rz = w1[2] - (w2[2] + shift[2])
    cdef cnp.float64_t norm = np.sqrt(rx*rx + ry*ry + rz*rz)

    cdef cnp.float64_t dvx, dvy, dvz, result

    if norm==0.0:
        result1[0] = 0.0 #radial velocity
        result2[0] = 0.0 #unused value
        result3[0] = 1.0 #number of pairs
    else:
       #calculate the difference velocity.
       dvx = (w1[3] - w2[3])
       dvy = (w1[4] - w2[4])
       dvz = (w1[5] - w2[5])

       #the radial component of the velocity difference
       # Since rx, ry, rz have a flipped sign for PBC case,
       #    the following definition requires no further modification
       result = (dvx*rx + dvy*ry + dvz*rz)/norm

       result1[0] = result #radial velocity
       result2[0] = 0.0 #unused value
       result3[0] = 1.0 #number of pairs



cdef void radial_velocity_variance_counter_weights(cnp.float64_t* w1,
                                                   cnp.float64_t* w2,
                                                   cnp.float64_t* shift,
                                                   cnp.float64_t* result1,
                                                   cnp.float64_t* result2,
                                                   cnp.float64_t* result3):
    """
    Calculate the relative radial velocity between two points minus an offset, and the
    squared quantity.  This function is used to calculate the variance using the
    "shifted data" technique where a constant value is subtracted from the value.

    func ID=2

    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0:3] x,y,z positions
        w1[3:6] vx, vy, vz velocities
        w1[6] offset

    w2 : pointer to an array
        weights array associated with data2
        w2[0:3] x,y,z positions
        w2[3:6] vx, vy, vz velocities
        w2[6] offset

    shift : pointer to an array
        Length-3 array storing the amount the points were shifted in each spatial
        dimension.  This is used when doing pair counts on periodic boxes and the
        points have been preshifted.

    result1 : pointer to a double
        relative radial velocity minus an offset

    result2 : pointer to a double
        relative radial velocity minus an offset squared

    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)

    """

    #calculate radial vector between points
    cdef cnp.float64_t rx = w1[0] - (w2[0] + shift[0])
    cdef cnp.float64_t ry = w1[1] - (w2[1] + shift[1])
    cdef cnp.float64_t rz = w1[2] - (w2[2] + shift[2])
    cdef cnp.float64_t norm = np.sqrt(rx*rx + ry*ry + rz*rz)

    cdef cnp.float64_t dvx, dvy, dvz, result

    if norm==0:
        result1[0] = 0.0
        result2[0] = 0.0
        result3[0] = 0.0
    else:
        #calculate the difference velocity.
        dvx = (w1[3] - w2[3])
        dvy = (w1[4] - w2[4])
        dvz = (w1[5] - w2[5])

        #the radial component of the velocity difference
        result = (dvx*rx + dvy*ry + dvz*rz)/norm - w1[6]*w2[6]

        result1[0] = result #radial velocity
        result2[0] = result*result #radial velocity squared
        result3[0] = 1.0 #number of pairs


cdef void relative_los_velocity_weights(cnp.float64_t* w1,
                                        cnp.float64_t* w2,
                                        cnp.float64_t* shift,
                                        cnp.float64_t* result1,
                                        cnp.float64_t* result2,
                                        cnp.float64_t* result3):
    """
    Calculate the relative line-of-sight (LOS) velocity between two points.

    func ID=3

    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0] vz velocities

    w2 : pointer to an array
        weights array associated with data2
        w2[0] vz velocities

    shift : pointer to an array
        Length-3 array storing the amount the points were shifted in each spatial
        dimension.  This is used when doing pair counts on periodic boxes and the
        points have been preshifted.

    result1 : pointer to a double
        relative LOS velocity

    result2 : pointer to a double
        0.0 (dummy)

    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)

    """
    #calculate radial vector between points
    # Note that due to the application of the shift,
    #   when PBCs are applied, rx, ry, rz has its normal sign flipped
    cdef cnp.float64_t rz = w1[2] - (w2[2] + shift[2])
    cdef cnp.float64_t norm = abs(rz)

    cdef cnp.float64_t dvz

    if rz == 0:
        dvz = -abs(w1[5] - w2[5])
    else:
        dvz = (w1[5] - w2[5])*rz/norm

    result1[0] = dvz #LOS velocity
    result2[0] = 0.0 #unused value
    result3[0] = 1.0 #number of pairs


cdef void los_velocity_variance_counter_weights(cnp.float64_t* w1,
                                                cnp.float64_t* w2,
                                                cnp.float64_t* shift,
                                                cnp.float64_t* result1,
                                                cnp.float64_t* result2,
                                                cnp.float64_t* result3):
    """
    Calculate the relative LOS velocity between two points minus an offset, and the
    squared quantity.  This function is used to calculate the variance using the
    "shifted data" technique where a constant value is subtracted from the value.

    func ID=4

    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0] vz velocities

    w2 : pointer to an array
        weights array associated with data2
        w2[0] vz velocities

    shift : pointer to an array
        Length-3 array storing the amount the points were shifted in each spatial
        dimension.  This is used when doing pair counts on periodic boxes and the
        points have been preshifted.

    result1 : pointer to a double
        relative LOS velocity minus an offset

    result2 : pointer to a double
        relative LOS velocity minus an offset squared

    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)

    """
    #calculate radial vector between points
    # Note that due to the application of the shift,
    #   when PBCs are applied, rx, ry, rz has its normal sign flipped
    cdef cnp.float64_t rz = w1[2] - (w2[2] + shift[2])
    cdef cnp.float64_t norm = abs(rz)

    cdef cnp.float64_t dvz

    if rz == 0:
        dvz = -abs(w1[5] - w2[5]) - w1[6]*w2[6]
    else:
        dvz = (w1[5] - w2[5])*rz/norm - w1[6]*w2[6]

    result1[0] = dvz #radial velocity
    result2[0] = dvz*dvz #radial velocity squared
    result3[0] = 1.0 #number of pairs
