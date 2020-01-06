# cython: language_level=2

import numpy as np
cimport numpy as np

#####built in weighting functions####

cdef double custom_func(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)

