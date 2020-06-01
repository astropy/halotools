# cython: language_level=2
cimport numpy as cnp

cdef double periodic_square_distance(
    cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
    cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2,
    cnp.float64_t* period)

cdef double square_distance(
    cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
    cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2)

cdef double perp_square_distance(
    cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t x2, cnp.float64_t y2)

cdef double para_square_distance(cnp.float64_t z1, cnp.float64_t z2)

cdef double periodic_perp_square_distance(
    cnp.float64_t x1, cnp.float64_t y1,
    cnp.float64_t x2, cnp.float64_t y2,
    cnp.float64_t* period)

cdef double periodic_para_square_distance(
    cnp.float64_t z1, cnp.float64_t z2, cnp.float64_t* period)

