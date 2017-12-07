cimport numpy as cnp

##### built-in positional weighting functions####

cdef cnp.float64_t pos_shape_dot_product_func(cnp.float64_t* w1, cnp.float64_t* w2, x1, y1, z1, x2, y2, z2)