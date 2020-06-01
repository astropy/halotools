# cython: language_level=2
cimport numpy as cnp

##### built-in weighting functions####

cdef cnp.float64_t mweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t sweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t eqweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t ineqweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t gweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t lweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t tgweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t tlweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t tweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t exweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t ratio_weights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t dotweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t squareddotweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t projdotweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t projsquareddotweights(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t squareddotweights_eq(cnp.float64_t* w1, cnp.float64_t* w2)
cdef cnp.float64_t squareddotweights_ineq(cnp.float64_t* w1, cnp.float64_t* w2)
