# cython: language_level=2
cimport numpy as cnp

#####built in weighting functions####

#radial functions
cdef void relative_radial_velocity_weights(cnp.float64_t* w1, cnp.float64_t* w2, cnp.float64_t* shift, cnp.float64_t* result1, cnp.float64_t* result2, cnp.float64_t* result3)
cdef void radial_velocity_variance_counter_weights(cnp.float64_t* w1, cnp.float64_t* w2, cnp.float64_t* shift, cnp.float64_t* result1, cnp.float64_t* result2, cnp.float64_t* result3)

#line-of-sight functions
cdef void relative_los_velocity_weights(cnp.float64_t* w1, cnp.float64_t* w2, cnp.float64_t* shift, cnp.float64_t* result1, cnp.float64_t* result2, cnp.float64_t* result3)
cdef void los_velocity_variance_counter_weights(cnp.float64_t* w1, cnp.float64_t* w2, cnp.float64_t* shift, cnp.float64_t* result1, cnp.float64_t* result2, cnp.float64_t* result3)
