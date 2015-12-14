
import numpy as np
cimport numpy as np

#####built in weighting functions####

#radial functions
cdef void relative_radial_velocity_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift, double *result1, double *result2, double *result3)
cdef void radial_velocity_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift, double *result1, double *result2, double *result3)
cdef void radial_velocity_variance_counter_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift, double *result1, double *result2, double *result3)

#line-of-sight functions
cdef void relative_los_velocity_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift, double *result1, double *result2, double *result3)
cdef void los_velocity_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift, double *result1, double *result2, double *result3)
cdef void los_velocity_variance_counter_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift, double *result1, double *result2, double *result3)
