# cython: language_level=2

cimport numpy as cnp

cdef class Condition:
    cdef bint func(self, cnp.int64_t, cnp.int64_t)

cdef class AlwaysTrue(Condition):
    cdef bint func(self, cnp.int64_t, cnp.int64_t)

cdef class MassFrac(Condition):
    cdef cnp.float64_t[:] mass1
    cdef cnp.float64_t[:] mass2
    cdef float fmin
    cdef float fmax
    cdef bint lower_equality
    cdef bint upper_equality

    cdef bint func(self, cnp.int64_t, cnp.int64_t)
