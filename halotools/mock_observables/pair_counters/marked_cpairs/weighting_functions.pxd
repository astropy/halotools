
import numpy as np
cimport numpy as np

#####built in weighting functions####

cdef double mweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double sweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double eqweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double ineqweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)

cdef double gweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double lweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)

cdef double tgweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double tlweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)

cdef double tweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double exweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)

cdef double radial_velocity_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double velocity_dot_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
cdef double velocity_angle_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)

