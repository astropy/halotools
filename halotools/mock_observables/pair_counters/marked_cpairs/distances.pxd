# cython: language_level=2
import numpy as np
cimport numpy as np

cdef double periodic_square_distance(np.float64_t x1,\
                                     np.float64_t y1,\
                                     np.float64_t z1,\
                                     np.float64_t x2,\
                                     np.float64_t y2,\
                                     np.float64_t z2,\
                                     np.float64_t* period)

cdef double square_distance(np.float64_t x1, np.float64_t y1, np.float64_t z1,\
                            np.float64_t x2, np.float64_t y2, np.float64_t z2)

cdef double perp_square_distance(np.float64_t x1, np.float64_t y1,\
                                 np.float64_t x2, np.float64_t y2)

cdef double para_square_distance(np.float64_t z1, np.float64_t z2)

cdef double periodic_perp_square_distance(np.float64_t x1, np.float64_t y1,\
                                          np.float64_t x2, np.float64_t y2,\
                                          np.float64_t* period)

cdef double periodic_para_square_distance(np.float64_t z1, np.float64_t z2,\
                                          np.float64_t* period)

