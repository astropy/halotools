
import numpy as np
cimport numpy as np

#####built in weighting functions####

cdef double custom_func(np.float64_t* w1, np.float64_t* w2)

cdef double mweights(np.float64_t* w1, np.float64_t* w2)
cdef double sweights(np.float64_t* w1, np.float64_t* w2)
cdef double eqweights(np.float64_t* w1, np.float64_t* w2)

cdef double gweights(np.float64_t* w1, np.float64_t* w2)
cdef double lweights(np.float64_t* w1, np.float64_t* w2)

cdef double tgweights(np.float64_t* w1, np.float64_t* w2)
cdef double tlweights(np.float64_t* w1, np.float64_t* w2)

cdef double tweights(np.float64_t* w1, np.float64_t* w2)
cdef double exweights(np.float64_t* w1, np.float64_t* w2)

