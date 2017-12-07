cimport numpy as cnp

##### built-in positional weighting functions####

cdef cnp.float64_t pos_shape_dot_product_func(cnp.float64_t* w1, cnp.float64_t* w2,
      cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
      cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2)
