# cython: profile=False

"""
weighting fuctions that return pairwise velocity calculations.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

__author__ = ["Duncan Campbell"]


cdef void relative_radial_velocity_weights(np.float64_t* w1,
                                  np.float64_t* w2,
                                  np.float64_t* shift,
                                  double* result1,
                                  double* result2,
                                  double* result3):
    """
    Calculate the relative radial velocity between two points.
    
    Return the relative radial velocity
    
    weighting vector (w1 w2) is:
    w[0:2] is the position vector x,y,z
    w[3:] is the velocity vector vx, vy, vz
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
    
    Return the radial multiplied radial velocities
    
    weighting vector (w1 w2) is:
    w[0:2] is the position vector x,y,z
    w[3:] is the velocity vector vx, vy, vz
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


cdef void velocity_variance_counter_weights(np.float64_t* w1,
                                            np.float64_t* w2,
                                            np.float64_t* shift,
                                            double* result1,
                                            double* result2,
                                            double* result3):
    """
    Calculate the relative radial velocity between two points, minus a value (a shift)
    
    Return the shifted relative radial velocity and the shited radial velocity squared
    
    weighting vector (w1 w2) is:
    w[0:2] is the position vector x,y,z
    w[3:] is the velocity vector vx, vy, vz
    w[6] is the shit value
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


