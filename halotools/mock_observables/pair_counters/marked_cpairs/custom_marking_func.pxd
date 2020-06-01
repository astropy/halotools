# cython: language_level=2
cimport numpy as cnp

##### declaration of user-defined custom marking function ####

cdef cnp.float64_t custom_func(cnp.float64_t* w1, cnp.float64_t* w2)

