  # cython: language_level=2
cimport numpy as cnp

##### built-in projected positional weighting functions####

cdef cnp.float64_t pos_shape_dot_product_func(cnp.float64_t* w1, cnp.float64_t* w2,
      cnp.float64_t x1, cnp.float64_t y1,
      cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t dxy_sq)

cdef cnp.float64_t gamma_plus_func(cnp.float64_t* w1, cnp.float64_t* w2,
      cnp.float64_t x1, cnp.float64_t y1,
      cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t dxy_sq)

cdef cnp.float64_t gamma_cross_func(cnp.float64_t* w1, cnp.float64_t* w2,
      cnp.float64_t x1, cnp.float64_t y1,
      cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t dxy_sq)

cdef cnp.float64_t squareddot_func(cnp.float64_t* w1, cnp.float64_t* w2,
      cnp.float64_t x1, cnp.float64_t y1,
      cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t dxy_sq)

cdef cnp.float64_t gamma_gamma_plus_func(cnp.float64_t* w1, cnp.float64_t* w2,
      cnp.float64_t x1, cnp.float64_t y1,
      cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t dxy_sq)

cdef cnp.float64_t gamma_gamma_cross_func(cnp.float64_t* w1, cnp.float64_t* w2,
      cnp.float64_t x1, cnp.float64_t y1,
      cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t dxy_sq)