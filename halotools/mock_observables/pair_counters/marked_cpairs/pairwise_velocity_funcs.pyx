# cython: profile=False

"""
weighting fuctions that return pairwise velocity calculations.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

__all__= ["relative_radial_velocity_weights", "radial_velocity_weights",\
          "radial_velocity_variance_counter_weights",\
          "relative_los_velocity_weights", "los_velocity_weights",\
          "los_velocity_variance_counter_weights"]
__author__ = ["Duncan Campbell"]


cdef void relative_radial_velocity_weights(np.float64_t* w1,
                                           np.float64_t* w2,
                                           np.float64_t* shift,
                                           double* result1,
                                           double* result2,
                                           double* result3):
    """
    Calculate the relative radial velocity between two points.
    
    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0:2] x,y,z positions
        w1[3:6] vx, vy, vz velocities
    
    w2 : pointer to an array
        weights array associated with data2
        w2[0:2] x,y,z positions
        w2[3:6] vx, vy, vz velocities
    
    shift : pointer to an array
        Legnth-3 array storing the amount the points were shifted in each spatial 
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
    cdef float rx = w1[0] - (w2[0] + shift[0])
    cdef float ry = w1[1] - (w2[1] + shift[1])
    cdef float rz = w1[2] - (w2[2] + shift[2])
    cdef float norm = sqrt(rx*rx + ry*ry + rz*rz)
    
    #if shift[i]<0 or shift[i]>0 return -1, else return 1
    cdef float xshift = -1.0*(shift[0]!=0.0) + (shift[0]==0.0)
    cdef float yshift = -1.0*(shift[1]!=0.0) + (shift[0]==0.0)
    cdef float zshift = -1.0*(shift[2]!=0.0) + (shift[0]==0.0)
    
    cdef float dvx, dvy, dvz, result
    
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
       result = (xshift*dvx*rx + yshift*dvy*ry + zshift*dvz*rz)/norm
       
       result1[0] = result #radial velocity
       result2[0] = 0.0 #unused value
       result3[0] = 1.0 #number of pairs


cdef void radial_velocity_weights(np.float64_t* w1,
                                  np.float64_t* w2,
                                  np.float64_t* shift,
                                  double* result1,
                                  double* result2,
                                  double* result3):
    """
    Calculate the radial velocity between two points.
    
    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0:2] x,y,z positions
        w1[3:6] vx, vy, vz velocities
    
    w2 : pointer to an array
        weights array associated with data2
        w2[0:2] x,y,z positions
        w2[3:6] vx, vy, vz velocities
    
    shift : pointer to an array
        Legnth-3 array storing the amount the points were shifted in each spatial 
        dimension.  This is used when doing pair counts on periodic boxes and the 
        points have been preshifted.
    
    result1 : pointer to a double
        radial velocity, v_r1 * v_r2
        
    result2 : pointer to a double
        0.0 (dummy)
        
    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)
    
    """
    
    #calculate radial vector between points
    cdef float rx = w1[0] - (w2[0] + shift[0])
    cdef float ry = w1[1] - (w2[1] + shift[1])
    cdef float rz = w1[2] - (w2[2] + shift[2])
    cdef float norm = sqrt(rx*rx + ry*ry + rz*rz)
    
    #if shift[i]<0 or shift[i]>0 return -1, else return 1
    cdef float xshift = -1.0*(shift[0]!=0.0) + (shift[0]==0.0)
    cdef float yshift = -1.0*(shift[1]!=0.0) + (shift[1]==0.0)
    cdef float zshift = -1.0*(shift[2]!=0.0) + (shift[2]==0.0)
    
    cdef float dvx, dvy, dvz, resulta, resultb
    
    if norm==0:
        result1[0] = 0.0 #radial velocity
        result2[0] = 0.0 #unused value
        result3[0] = 1.0 #number of pairs
    else:        
       #the radial component of the velocities
       resulta = (xshift*w1[3]*rx + yshift*w1[4]*ry + zshift*w1[5]*rz)/norm
       resultb = (xshift*w2[3]*rx + yshift*w2[4]*ry + zshift*w2[5]*rz)/norm
       
       result1[0] = resulta*resultb #radial velocity
       result2[0] = 0.0 #unused value
       result3[0] = 1.0 #number of pairs


cdef void radial_velocity_variance_counter_weights(np.float64_t* w1,
                                                   np.float64_t* w2,
                                                   np.float64_t* shift,
                                                   double* result1,
                                                   double* result2,
                                                   double* result3):
    """
    Calculate the relative radial velocity between two points minus an offset, and the 
    squared quantity.  This function is used to calculate the variance using the 
    "shifted data" technique where a constant value is subtracted from the value.
    
    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0:2] x,y,z positions
        w1[3:6] vx, vy, vz velocities
        w1[7] offset
    
    w2 : pointer to an array
        weights array associated with data2
        w2[0:2] x,y,z positions
        w2[3:6] vx, vy, vz velocities
        w2[7] offset
    
    shift : pointer to an array
        Legnth-3 array storing the amount the points were shifted in each spatial 
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
    cdef float rx = w1[0] - (w2[0] + shift[0])
    cdef float ry = w1[1] - (w2[1] + shift[1])
    cdef float rz = w1[2] - (w2[2] + shift[2])
    cdef float norm = sqrt(rx*rx + ry*ry + rz*rz)
    
    #if shift[i]<0 or shift[i]>0 return -1, else return 1
    cdef float xshift = -1.0*(shift[0]!=0.0) + (shift[0]==0.0)
    cdef float yshift = -1.0*(shift[1]!=0.0) + (shift[1]==0.0)
    cdef float zshift = -1.0*(shift[2]!=0.0) + (shift[2]==0.0)
    
    cdef float dvx, dvy, dvz, result
    
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
        result = (xshift*dvx*rx + yshift*dvy*ry + zshift*dvz*rz)/norm - w1[6]*w2[6]
        
        result1[0] = result #radial velocity
        result2[0] = result*result #radial velocity squared
        result3[0] = 1.0 #number of pairs


cdef void relative_los_velocity_weights(np.float64_t* w1,
                                        np.float64_t* w2,
                                        np.float64_t* shift,
                                        double* result1,
                                        double* result2,
                                        double* result3):
    """
    Calculate the relative line-of-sight (LOS) velocity between two points.
    
    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0] vz velocities
    
    w2 : pointer to an array
        weights array associated with data2
        w2[0] vz velocities
    
    shift : pointer to an array
        Legnth-3 array storing the amount the points were shifted in each spatial 
        dimension.  This is used when doing pair counts on periodic boxes and the 
        points have been preshifted.
    
    result1 : pointer to a double
        relative LOS velocity
        
    result2 : pointer to a double
        0.0 (dummy)
        
    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)
    
    """
    
    cdef float dvz = fabs(w1[0] - w2[0])
    result1[0] = dvz #LOS velocity
    result2[0] = 0.0 #unused value
    result3[0] = 1.0 #number of pairs


cdef void los_velocity_weights(np.float64_t* w1,
                               np.float64_t* w2,
                               np.float64_t* shift,
                               double* result1,
                               double* result2,
                               double* result3):
    """
    Calculate the line-of-sight (LOS) velocity between two points.
    
    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0] vz velocities
    
    w2 : pointer to an array
        weights array associated with data2
        w2[0] vz velocities
    
    shift : pointer to an array
        Legnth-3 array storing the amount the points were shifted in each spatial 
        dimension.  This is used when doing pair counts on periodic boxes and the 
        points have been preshifted.
    
    result1 : pointer to a double
        LOS velocity v_z1 * v_z2
        
    result2 : pointer to a double
        0.0 (dummy)
        
    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)
    
    """
    
    result1[0] = w1[0]*w2[0] #LOS velocity
    result2[0] = 0.0 #unused value
    result3[0] = 1.0 #number of pairs


cdef void los_velocity_variance_counter_weights(np.float64_t* w1,
                                                np.float64_t* w2,
                                                np.float64_t* shift,
                                                double* result1,
                                                double* result2,
                                                double* result3):
    """
    Calculate the relative LOS velocity between two points minus an offset, and the 
    squared quantity.  This function is used to calculate the variance using the 
    "shifted data" technique where a constant value is subtracted from the value.
    
    Parameters
    ----------
    w1 : pointer to an array
        weights array associated with data1.
        w1[0] vz velocities
    
    w2 : pointer to an array
        weights array associated with data2
        w2[0] vz velocities
    
    shift : pointer to an array
        Legnth-3 array storing the amount the points were shifted in each spatial 
        dimension.  This is used when doing pair counts on periodic boxes and the 
        points have been preshifted.
    
    result1 : pointer to a double
        relative LOS velocity minus an offset
        
    result2 : pointer to a double
        relative LOS velocity minus an offset squared
        
    result3 : pointer to a double
        1.0 (pairs involved, but also kind-of a dummy)
    
    """
    
    cdef float dvz = fabs(w1[0] - w2[0]) - w1[1]*w2[1]
    
    result1[0] = dvz #LOS velocity
    result2[0] = dvz*dvz #LOS velocity squared
    result3[0] = 1.0 #number of pairs

