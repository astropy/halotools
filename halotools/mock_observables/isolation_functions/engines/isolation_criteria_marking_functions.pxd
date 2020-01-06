# cython: language_level=2
cimport numpy as cnp

cdef bint trivial(cnp.float64_t* w1, cnp.float64_t* w2)
cdef bint gt_cond(cnp.float64_t* w1, cnp.float64_t* w2)
cdef bint lt_cond(cnp.float64_t* w1, cnp.float64_t* w2)
cdef bint eq_cond(cnp.float64_t* w1, cnp.float64_t* w2)
cdef bint neq_cond(cnp.float64_t* w1, cnp.float64_t* w2)
cdef bint tg_cond(cnp.float64_t* w1, cnp.float64_t* w2)
cdef bint lg_cond(cnp.float64_t* w1, cnp.float64_t* w2)



