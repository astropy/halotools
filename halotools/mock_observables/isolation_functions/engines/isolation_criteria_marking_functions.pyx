# cython: language_level=2
# cython: profile=False
""" Module containing C implementations of the isolation functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
cimport numpy as np

__author__ = ["Duncan Campbell", "Andrew Hearin"]
__all__ = ('trivial', 'gt_cond', 'lt_cond', 'eq_cond', 'neq_cond', 'tg_cond', 'lg_cond')

cdef bint trivial(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    0
    """
    cdef bint result = 1
    return result

cdef bint gt_cond(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    1
    """
    cdef bint result
    result = (w1[0]>w2[0])
    return result

cdef bint lt_cond(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    2
    """
    cdef bint result
    result = (w1[0]<w2[0])
    return result

cdef bint eq_cond(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    3
    """
    cdef bint result
    result = (w1[0]==w2[0])
    return result

cdef bint neq_cond(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    4
    """
    cdef bint result
    result = (w1[0]!=w2[0])
    return result

cdef bint tg_cond(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    5
    """
    cdef bint result
    result = (w1[0]>(w2[0]+w1[1]))
    return result

cdef bint lg_cond(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    6
    """
    cdef bint result
    result = (w1[0]<(w2[0]+w1[1]))
    return result

