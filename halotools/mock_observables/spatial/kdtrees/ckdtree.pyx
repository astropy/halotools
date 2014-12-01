# cython: profile=True
# filename: ckdtree.pyx
# cython: boundscheck=False, overflowcheck=False

# Copyright Anne M. Archibald 2008
# Additional contributions by Patrick Varilly and Sturla Molden
# Released under the scipy license

# partial support for periodic boundary conditions implemented by Stephen Skory 2011
# yt-project
# https://bitbucket.org/yt_analysis/yt

# Modified by Duncan Campbell to fully include periodic boundary conditions.
# Added functions to use weights for counting pairs
# Yale University
# july 22, 2014

import numpy as np
import scipy.sparse

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

cdef extern from "limits.h":
    long LONG_MAX
cdef np.float64_t infinity = np.inf

__all__ = ['cKDTree']


# Notes on int and 64-bit cleanliness
# ===================================
#
# Never use a bare "int" for array indices; use np.intp_t instead.  A Python
# int and np.int is a C long on Python 2.x, which can be 32 bits on 64-bit
# systems (e.g. Windows).
#
# The exception is as the return type of a nominally void function, which
# instead returns 0 and signals a Python exception by returning -1.
#
# Also, when converting np.intp_t's to Python, you should explicitly cast to
# a Python "int" object if sizeof(long) < sizeof(np.intp_t).  From the
# mailing list (Sturla Molden): "On Win 64 we should use Python long instead
# of Python int if a C long (i.e. Python int) overflows, which the function
# int() will ensure.  Cython automatically converts np.npy_intp [==
# np.intp_t] to Python long on Win 64, which we want to convert to a Python
# int if it is possible.  On other platforms we don't want this extra
# overhead."

# The following utility functions help properly add int tuples to sets and
# ints to lists.  The results of the if is known at compile time, so the
# test is optimized away.

cdef inline int set_add_pair(set results, np.intp_t i, np.intp_t j) except -1:

    if sizeof(long) < sizeof(np.intp_t):
        # Win 64
        results.add((int(i), int(j)))
    else:
        # Other platforms
        results.add((i, j))
    return 0

cdef inline int set_add_ordered_pair(set results, np.intp_t i, np.intp_t j) except -1:

    if sizeof(long) < sizeof(np.intp_t):
        # Win 64
        if i < j:
            results.add((int(i), int(j)))
        else:
            results.add((int(j), int(i)))
    else:
        # Other platforms
        if i < j:
            results.add((i, j))
        else:
            results.add((j, i))
    return 0

cdef inline int list_append(list results, np.intp_t i) except -1:
    if sizeof(long) < sizeof(np.intp_t):
        # Win 64
        if i <= <np.intp_t>LONG_MAX:  # CHECK COMPARISON DIRECTION
            results.append(int(i))
        else:
            results.append(i)
    else:
        # Other platforms
        results.append(i)
    return 0


# Priority queue
# ==============
cdef union heapcontents:    # FIXME: Unions are not always portable, verify this 
    np.intp_t intdata     # union is never used in an ABI dependent way.
    char* ptrdata

cdef struct heapitem:
    np.float64_t priority
    heapcontents contents

cdef class heap(object):
    cdef np.intp_t n
    cdef heapitem* heap
    cdef np.intp_t space
    
    def __init__(heap self, np.intp_t initial_size):
        cdef void *tmp
        self.space = initial_size
        self.heap = <heapitem*> NULL
        tmp = stdlib.malloc(sizeof(heapitem)*self.space)
        if tmp == NULL:
            raise MemoryError
        self.heap = <heapitem*> tmp  
        self.n = 0

    def __dealloc__(heap self):
        if self.heap != <heapitem*> NULL:
            stdlib.free(self.heap)

    cdef inline int _resize(heap self, np.intp_t new_space) except -1:
        cdef void *tmp
        if new_space < self.n:
            raise ValueError("Heap containing %d items cannot be resized to %d" % (int(self.n), int(new_space)))
        self.space = new_space
        tmp = stdlib.realloc(<void*>self.heap, new_space*sizeof(heapitem))
        if tmp == NULL:
            raise MemoryError
        self.heap = <heapitem*> tmp
        return 0

    @cython.cdivision(True)
    cdef inline int push(heap self, heapitem item) except -1:
        cdef np.intp_t i
        cdef heapitem t

        self.n += 1
        if self.n > self.space:
            self._resize(2 * self.space + 1)
            
        i = self.n - 1
        self.heap[i] = item
        
        while i > 0 and self.heap[i].priority < self.heap[(i - 1) // 2].priority:
            t = self.heap[(i - 1) // 2]
            self.heap[(i - 1) // 2] = self.heap[i]
            self.heap[i] = t
            i = (i - 1) // 2
        return 0
    
    
    cdef heapitem peek(heap self):
        return self.heap[0]
    
    
    @cython.cdivision(True)
    cdef int remove(heap self) except -1:
        cdef heapitem t
        cdef np.intp_t i, j, k, l
    
        self.heap[0] = self.heap[self.n-1]
        self.n -= 1
        # No point in freeing up space as the heap empties.
        # The whole heap gets deallocated at the end of any query below
        #if self.n < self.space//4 and self.space>40: #FIXME: magic number
        #    self._resize(self.space // 2 + 1)
        i=0
        j=1
        k=2
        while ((j<self.n and 
                    self.heap[i].priority > self.heap[j].priority or
                k<self.n and 
                    self.heap[i].priority > self.heap[k].priority)):
            if k<self.n and self.heap[j].priority>self.heap[k].priority:
                l = k
            else:
                l = j
            t = self.heap[l]
            self.heap[l] = self.heap[i]
            self.heap[i] = t
            i = l
            j = 2*i+1
            k = 2*i+2
        return 0
    
    cdef int pop(heap self, heapitem *it) except -1:
        it[0] = self.peek()
        self.remove()
        return 0


# Utility functions
# =================
cdef inline np.float64_t dmax(np.float64_t x, np.float64_t y):
    if x>y:
        return x
    else:
        return y

cdef inline np.float64_t dabs(np.float64_t x):
    if x>0:
        return x
    else:
        return -x

cdef inline np.float64_t dmin(np.float64_t x, np.float64_t y):
    if x<y:
        return x
    else:
        return y

cdef inline np.int_t imin(np.int_t x, np.int_t y):
    if x<y:
        return x
    else:
        return y

cdef inline np.int_t imax(np.int_t x, np.int_t y):
    if x>y:
        return x
    else:
        return y


# Utility for building a coo matrix incrementally
cdef class coo_entries:
    cdef:
        np.intp_t n, n_max
        np.ndarray i, j
        np.ndarray v
        np.intp_t *i_data
        np.intp_t *j_data
        np.float64_t *v_data
    
    def __init__(self):
        self.n = 0
        self.n_max = 10
        self.i = np.empty(self.n_max, dtype=np.intp)
        self.j = np.empty(self.n_max, dtype=np.intp)
        self.v = np.empty(self.n_max, dtype=np.float64)
        self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
        self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
        self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)

    cdef void add(coo_entries self, np.intp_t i, np.intp_t j, np.float64_t v):
        cdef np.intp_t k
        if self.n == self.n_max:
            self.n_max *= 2
            self.i.resize(self.n_max)
            self.j.resize(self.n_max)
            self.v.resize(self.n_max)
            self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
            self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
            self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)
        k = self.n
        self.i_data[k] = i
        self.j_data[k] = j
        self.v_data[k] = v
        self.n += 1

    def to_matrix(coo_entries self, shape=None):
        # Shrink arrays to size
        self.i.resize(self.n)
        self.j.resize(self.n)
        self.v.resize(self.n)
        self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
        self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
        self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)
        self.n_max = self.n
        return scipy.sparse.coo_matrix((self.v, (self.i, self.j)), shape=shape, dtype=np.float64)



# Measuring distances
# ===================
cdef inline np.float64_t _distance_p(np.float64_t *x, np.float64_t *y,
                                     np.float64_t p, np.intp_t k,
                                     np.float64_t upperbound):
    """Compute the distance between x and y

    Computes the Minkowski p-distance to the power p between two points.
    If the distance**p is larger than upperbound, then any number larger
    than upperbound may be returned (the calculation is truncated).
    """
    cdef np.intp_t i
    cdef np.float64_t r, z
    r = 0
    if p==2:
        for i in range(k):
            z = x[i] - y[i]
            r += z*z
            if r>upperbound:
                return r 
    elif p==infinity:
        for i in range(k):
            r = dmax(r,dabs(x[i]-y[i]))
            if r>upperbound:
                return r
    elif p==1:
        for i in range(k):
            r += dabs(x[i]-y[i])
            if r>upperbound:
                return r
    else:
        for i in range(k):
            r += dabs(x[i]-y[i])**p
            if r>upperbound:
                return r
    return r


cdef inline np.float64_t _distance_p_periodic(np.float64_t *x, np.float64_t *y,
                                              np.float64_t p, np.intp_t k,
                                              np.float64_t upperbound,
                                              np.float64_t *period):
    """Compute the distance between x and y using periodic boundary conditoons.
    
    Computes the Minkowski p-distance to the power p between two points.
    If the distance**p is larger than upperbound, then any number larger
    than upperbound may be returned (the calculation is truncated).
    """
    
    cdef int i
    cdef np.float64_t r, m
    r = 0
    if p==infinity:
        for i in range(k):
            diff = dabs(x[i] - y[i])
            m = dmin(diff, period[i] - diff)
            r = dmax(r,m)
            if r>upperbound:
                return r
    elif p==1:
        for i in range(k):
            diff = dabs(x[i] - y[i])
            m = dmin(diff, period[i] - diff)
            r += m
            if r>upperbound:
                return r
    elif p==2:
        for i in range(k):
            diff = dabs(x[i] - y[i])
            m = dmin(diff, period[i] - diff)
            r += m*m
            if r>upperbound:
                return r
    else:
        for i in range(k):
            diff = dabs(x[i] - y[i])
            m = dmin(diff, period[i] - diff)
            r += m**p
            if r>upperbound:
                return r
    return r


cdef _projected_distance_p_periodic(np.float64_t *x, np.float64_t *y,
                                                        np.float64_t p, np.intp_t k,
                                                        np.float64_t *period,
                                                        np.float64_t *los):
    """
    compute the projected distances between points
    """
    
    cdef int i
    cdef np.float64_t r1, r2, d_para_x, d_para_y, d_perp_x, d_perp_y, d_para, d_perp
    r1 = 0
    r2 = 0
    if p==2:
        for i in range(k):
            d_para_x = x[i]*los[i]
            d_perp_x = x[i]-x[i]*los[i]
            d_para_y = y[i]*los[i]
            d_perp_y = y[i]-y[i]*los[i]
            d_para = dabs(d_para_x-d_para_y)
            d_perp = dabs(d_perp_x-d_perp_y)
            d_para = dmin(d_para, period[i] - d_para)
            d_perp = dmin(d_perp, period[i] - d_perp)
            r1 += d_para*d_para
            r2 += d_perp*d_perp
    else:
        raise ValueError("only takes p==2 right now!")
                
    return (r1, r2)


# Interval arithmetic
# ===================

cdef class Rectangle:
    cdef np.intp_t m
    cdef np.float64_t *mins
    cdef np.float64_t *maxes
    cdef np.ndarray mins_arr, maxes_arr

    def __init__(self, mins_arr, maxes_arr):
        # Copy array data
        self.mins_arr = np.array(mins_arr, dtype=np.float64, order='C')
        self.maxes_arr = np.array(maxes_arr, dtype=np.float64, order='C')
        self.mins = <np.float64_t*>np.PyArray_DATA(self.mins_arr)
        self.maxes = <np.float64_t*>np.PyArray_DATA(self.maxes_arr)
        self.m = self.mins_arr.shape[0]

# 1-d pieces
# These should only be used if p != infinity
# DC added periodic versions of the functions but left the originals incase they need to
#   be used because they are faster.
cdef inline np.float64_t min_dist_point_interval_p(np.float64_t* x,
                                                   Rectangle rect,
                                                   np.intp_t k,
                                                   np.float64_t p):    
    """Compute the minimum distance along dimension k between x and
    a point in the hyperrectangle.
    """
    return dmax(0, dmax(rect.mins[k] - x[k], x[k] - rect.maxes[k])) ** p

cdef inline np.float64_t min_dist_point_interval_p_periodic(np.float64_t* x,
                                                   Rectangle rect,
                                                   np.intp_t k,
                                                   np.float64_t p,
                                                   np.float64_t *period):
    """Compute the minimum distance along dimension k between x and
    a point in the hyperrectangle.
    """
    d_left = dmin(dabs(rect.mins[k] - x[k]), period[k] - dabs(rect.mins[k] - x[k]))
    d_right = dmin(dabs(rect.maxes[k] - x[k]), period[k] - dabs(rect.maxes[k] - x[k]))
    result = dmin(d_left,d_right)
    if dmax(0, dmax(rect.mins[k] - x[k], x[k] - rect.maxes[k])) == 0: return 0
    else: return result ** p

cdef inline np.float64_t max_dist_point_interval_p(np.float64_t* x,
                                                   Rectangle rect,
                                                   np.intp_t k,
                                                   np.float64_t p):
    """Compute the maximum distance along dimension k between x and
    a point in the hyperrectangle.
    """
    return dmax(rect.maxes[k] - x[k], x[k] - rect.mins[k]) ** p

cdef inline np.float64_t max_dist_point_interval_p_periodic(np.float64_t* x,
                                                   Rectangle rect,
                                                   np.intp_t k,
                                                   np.float64_t p,
                                                   np.float64_t *period):
    """Compute the maximum distance along dimension k between x and
    a point in the hyperrectangle.
    """
    d_left = dmin(dabs(rect.mins[k] - x[k]), period[k] - dabs(rect.mins[k] - x[k]))
    d_right = dmin(dabs(rect.maxes[k] - x[k]), period[k] - dabs(rect.maxes[k] - x[k]))
    result = dmax(d_left,d_right)
    delta_1 = rect.maxes[k]-rect.mins[k]
    if delta_1 >= period[k]/2.0: return (period[k]/2.0) ** 2.0
    else: return result  ** p

cdef inline np.float64_t min_dist_interval_interval_p(Rectangle rect1,
                                                      Rectangle rect2,
                                                      np.intp_t k,
                                                      np.float64_t p):
    """Compute the minimum distance along dimension k between points in
    two hyperrectangles.
    """
    return dmax(0, dmax(rect1.mins[k] - rect2.maxes[k],
                        rect2.mins[k] - rect1.maxes[k])) ** p

cdef inline np.float64_t min_dist_interval_interval_p_periodic(Rectangle rect1,
                                                      Rectangle rect2,
                                                      np.intp_t k,
                                                      np.float64_t p,
                                                      np.float64_t *period):
    """Compute the minimum distance along dimension k between points in
    two hyperrectangles.
    """
    d_lr = dmin(dabs(rect1.mins[k] - rect2.maxes[k]), period[k] - dabs(rect1.mins[k] - rect2.maxes[k]))
    d_rl = dmin(dabs(rect1.maxes[k] - rect2.mins[k]), period[k] - dabs(rect1.maxes[k] - rect2.mins[k]))
    if dmax(0, dmax(rect1.mins[k] - rect2.maxes[k], rect2.mins[k] - rect1.maxes[k])) == 0: return 0 #overlap
    else: return dmin(d_lr, d_rl) ** p
    

cdef inline np.float64_t max_dist_interval_interval_p(Rectangle rect1,
                                                      Rectangle rect2,
                                                      np.intp_t k,
                                                      np.float64_t p):
    """Compute the maximum distance along dimension k between points in
    two hyperrectangles.
    """
    return dmax(rect1.maxes[k] - rect2.mins[k], rect2.maxes[k] - rect1.mins[k]) ** p

cdef inline np.float64_t max_dist_interval_interval_p_periodic(Rectangle rect1,
                                                      Rectangle rect2,
                                                      np.intp_t k,
                                                      np.float64_t p,
                                                      np.float64_t *period):
    """Compute the maximum distance along dimension k between points in
    two hyperrectangles.
    """
    d_lr = dmin(dabs(rect1.mins[k] - rect2.maxes[k]), period[k] - dabs(rect1.mins[k] - rect2.maxes[k]))
    d_rl = dmin(dabs(rect1.maxes[k] - rect2.mins[k]), period[k] - dabs(rect1.maxes[k] - rect2.mins[k]))
    d_rr = dmin(dabs(rect1.maxes[k] - rect2.maxes[k]), period[k] - dabs(rect1.maxes[k] - rect2.maxes[k]))
    d_ll = dmin(dabs(rect1.mins[k] - rect2.mins[k]), period[k] - dabs(rect1.mins[k] - rect2.mins[k]))
    delta_1 = rect1.maxes[k]-rect1.mins[k]
    delta_2 = rect2.maxes[k]-rect2.mins[k]
    if (delta_1+delta_2) >= (period[k]/2.0): return (period[k]/2.0) ** p
    else: return dmax(d_lr,dmax(d_rl,dmax(d_rr,d_ll))) ** p


#note: DC has not modified these to work with periodic boundary conditions despite the names!
#I don't think ill be using a p=inf metric anytime soon... -DC

# Interval arithmetic in m-D
# ==========================

# These should be used only for p == infinity
cdef inline np.float64_t min_dist_point_rect_p_inf(np.float64_t* x,
                                                   Rectangle rect):
    """Compute the minimum distance between x and the given hyperrectangle."""
    cdef np.intp_t i
    cdef np.float64_t min_dist = 0.
    for i in range(rect.m):
        min_dist = dmax(min_dist, dmax(rect.mins[i]-x[i], x[i]-rect.maxes[i]))
    return min_dist

cdef inline np.float64_t min_dist_point_rect_p_inf_periodic(np.float64_t* x,
                                                   Rectangle rect,
                                                   np.float64_t *period):
    """Compute the minimum distance between x and the given hyperrectangle."""
    cdef np.intp_t i
    cdef np.float64_t min_dist = 0.
    for i in range(rect.m):
        min_dist = dmax(min_dist, dmax(rect.mins[i]-x[i], x[i]-rect.maxes[i]))
    return min_dist

cdef inline np.float64_t max_dist_point_rect_p_inf(np.float64_t* x,
                                                   Rectangle rect):
    """Compute the maximum distance between x and the given hyperrectangle."""
    cdef np.intp_t i
    cdef np.float64_t max_dist = 0.
    for i in range(rect.m):
        max_dist = dmax(max_dist, dmax(rect.maxes[i]-x[i], x[i]-rect.mins[i]))
    return max_dist

cdef inline np.float64_t max_dist_point_rect_p_inf_periodic(np.float64_t* x,
                                                            Rectangle rect,
                                                            np.float64_t *period):
    """Compute the maximum distance between x and the given hyperrectangle."""
    cdef np.intp_t i
    cdef np.float64_t max_dist = 0.
    for i in range(rect.m):
        max_dist = dmax(max_dist, dmax(rect.maxes[i]-x[i], x[i]-rect.mins[i]))
    return max_dist

cdef inline np.float64_t min_dist_rect_rect_p_inf(Rectangle rect1,
                                                  Rectangle rect2):
    """Compute the minimum distance between points in two hyperrectangles."""
    cdef np.intp_t i
    cdef np.float64_t min_dist = 0.
    for i in range(rect1.m):
        min_dist = dmax(min_dist, dmax(rect1.mins[i] - rect2.maxes[i],
                                       rect2.mins[i] - rect1.maxes[i]))
    return min_dist

cdef inline np.float64_t min_dist_rect_rect_p_inf_periodic(Rectangle rect1,
                                                           Rectangle rect2,
                                                           np.float64_t *period):
    """Compute the minimum distance between points in two hyperrectangles."""
    cdef np.intp_t i
    cdef np.float64_t min_dist = 0.
    for i in range(rect1.m):
        min_dist = dmax(min_dist, dmax(rect1.mins[i] - rect2.maxes[i],
                                       rect2.mins[i] - rect1.maxes[i]))
    return min_dist

cdef inline np.float64_t max_dist_rect_rect_p_inf(Rectangle rect1,
                                                  Rectangle rect2):
    """Compute the maximum distance between points in two hyperrectangles."""
    cdef np.intp_t i
    cdef np.float64_t max_dist = 0.
    for i in range(rect1.m):
        max_dist = dmax(max_dist, dmax(rect1.maxes[i] - rect2.mins[i],
                                       rect2.maxes[i] - rect1.mins[i]))
    return max_dist

cdef inline np.float64_t max_dist_rect_rect_p_inf_periodic(Rectangle rect1,
                                                           Rectangle rect2,
                                                           np.float64_t *period):
    """Compute the maximum distance between points in two hyperrectangles."""
    cdef np.intp_t i
    cdef np.float64_t max_dist = 0.
    for i in range(rect1.m):
        max_dist = dmax(max_dist, dmax(rect1.maxes[i] - rect2.mins[i],
                                       rect2.maxes[i] - rect1.mins[i]))
    return max_dist


# Rectangle-to-rectangle distance tracker
# =======================================
#
# The logical unit that repeats over and over is to keep track of the
# maximum and minimum distances between points in two hyperrectangles
# as these rectangles are successively split.
#
# Example
# -------
# # node1 encloses points in rect1, node2 encloses those in rect2
#
# cdef RectRectDistanceTracker dist_tracker
# dist_tracker = RectRectDistanceTracker(rect1, rect2, p)
#
# ...
#
# if dist_tracker.min_distance < ...:
#     ...
#
# dist_tracker.push_less_of(1, node1)
# do_something(node1.less, dist_tracker)
# dist_tracker.pop()
#
# dist_tracker.push_greater_of(1, node1)
# do_something(node1.greater, dist_tracker)
# dist_tracker.pop()

cdef struct RR_stack_item:
    np.intp_t which
    np.intp_t split_dim
    double min_along_dim, max_along_dim
    np.float64_t min_distance, max_distance

cdef np.intp_t LESS = 1
cdef np.intp_t GREATER = 2

cdef class RectRectDistanceTracker(object):
    cdef Rectangle rect1, rect2
    cdef np.float64_t p, epsfac, upper_bound
    cdef np.float64_t min_distance, max_distance

    cdef np.intp_t stack_size, stack_max_size
    cdef RR_stack_item *stack
    cdef readonly np.ndarray cperiod

    # Stack handling
    cdef int _init_stack(self) except -1:
        cdef void *tmp
        self.stack_max_size = 10
        tmp = stdlib.malloc(sizeof(RR_stack_item) * self.stack_max_size)
        if tmp == NULL:
            raise MemoryError
        self.stack = <RR_stack_item*> tmp
        self.stack_size = 0
        return 0

    cdef int _resize_stack(self, np.intp_t new_max_size) except -1:
        cdef void *tmp
        self.stack_max_size = new_max_size
        tmp = stdlib.realloc(<RR_stack_item*> self.stack, new_max_size * sizeof(RR_stack_item))
        if tmp == NULL:
            raise MemoryError
        self.stack = <RR_stack_item*> tmp
        return 0
    
    cdef int _free_stack(self) except -1:
        if self.stack != <RR_stack_item*> NULL:
            stdlib.free(self.stack)
        return 0
    

    def __init__(self, Rectangle rect1, Rectangle rect2,
                 np.float64_t p, np.float64_t eps, np.float64_t upper_bound,
                 object period=None):
                 
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*rect1.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        self.cperiod = cperiod
        
        if rect1.m != rect2.m:
            raise ValueError("rect1 and rect2 have different dimensions")

        self.rect1 = rect1
        self.rect2 = rect2
        self.p = p
        
        # internally we represent all distances as distance ** p
        if p != infinity and upper_bound != infinity:
            self.upper_bound = upper_bound ** p
        else:
            self.upper_bound = upper_bound

        # fiddle approximation factor
        if eps == 0:
            self.epsfac = 1
        elif p == infinity:
            self.epsfac = 1 / (1 + eps)
        else:
            self.epsfac = 1 / (1 + eps) ** p

        self._init_stack()

        # Compute initial min and max distances
        if self.p == infinity:
            self.min_distance = min_dist_rect_rect_p_inf(rect1, rect2)
            self.max_distance = max_dist_rect_rect_p_inf(rect1, rect2)
        else:
            self.min_distance = 0.
            self.max_distance = 0.
            for i in range(rect1.m):
                #self.min_distance += min_dist_interval_interval_p(rect1, rect2, i, p)
                #self.max_distance += max_dist_interval_interval_p(rect1, rect2, i, p)
                self.min_distance += min_dist_interval_interval_p_periodic(rect1, rect2, i, p, <np.float64_t*>cperiod.data)
                self.max_distance += max_dist_interval_interval_p_periodic(rect1, rect2, i, p, <np.float64_t*>cperiod.data)

    def __dealloc__(self):
        self._free_stack()

    cdef int push(self, np.intp_t which, np.intp_t direction,
                  np.intp_t split_dim,
                  np.float64_t split_val) except -1:

        cdef Rectangle rect
        if which == 1:
            rect = self.rect1
        else:
            rect = self.rect2
            
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        cperiod = self.cperiod

        # Push onto stack
        if self.stack_size == self.stack_max_size:
            self._resize_stack(self.stack_max_size * 2)
            
        cdef RR_stack_item *item = &self.stack[self.stack_size]
        self.stack_size += 1
        item.which = which
        item.split_dim = split_dim
        item.min_distance = self.min_distance
        item.max_distance = self.max_distance
        item.min_along_dim = rect.mins[split_dim]
        item.max_along_dim = rect.maxes[split_dim]

        # Update min/max distances
        if self.p != infinity:
            #self.min_distance -= min_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)
            #self.max_distance -= max_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)
            self.min_distance -= min_dist_interval_interval_p_periodic(self.rect1, self.rect2, split_dim, self.p, <np.float64_t*>cperiod.data)
            self.max_distance -= max_dist_interval_interval_p_periodic(self.rect1, self.rect2, split_dim, self.p, <np.float64_t*>cperiod.data)

        if direction == LESS:
            rect.maxes[split_dim] = split_val
        else:
            rect.mins[split_dim] = split_val

        if self.p != infinity:
            #self.min_distance += min_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)
            #self.max_distance += max_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)
            self.min_distance += min_dist_interval_interval_p_periodic(self.rect1, self.rect2, split_dim, self.p, <np.float64_t*>cperiod.data)
            self.max_distance += max_dist_interval_interval_p_periodic(self.rect1, self.rect2, split_dim, self.p, <np.float64_t*>cperiod.data)
        else:
            self.min_distance = min_dist_rect_rect_p_inf(self.rect1, self.rect2)
            self.max_distance = max_dist_rect_rect_p_inf(self.rect1, self.rect2)
            
        return 0

    
    cdef inline int push_less_of(self, np.intp_t which,
                                 innernode *node) except -1:
        return self.push(which, LESS, node.split_dim, node.split)

    
    cdef inline int push_greater_of(self, np.intp_t which,
                                    innernode *node) except -1:
        return self.push(which, GREATER, node.split_dim, node.split)

    
    cdef inline int pop(self) except -1:
        # Pop from stack
        self.stack_size -= 1
        assert self.stack_size >= 0
        
        cdef RR_stack_item* item = &self.stack[self.stack_size]
        self.min_distance = item.min_distance
        self.max_distance = item.max_distance

        if item.which == 1:
            self.rect1.mins[item.split_dim] = item.min_along_dim
            self.rect1.maxes[item.split_dim] = item.max_along_dim
        else:
            self.rect2.mins[item.split_dim] = item.min_along_dim
            self.rect2.maxes[item.split_dim] = item.max_along_dim
        
        return 0

# Point-to-rectangle distance tracker
# ===================================
#
# The other logical unit that is used in query_ball_point is to keep track
# of the maximum and minimum distances between points in a hyperrectangle
# and another fixed point as the rectangle is successively split.
#
# Example
# -------
# # node encloses points in rect
#
# cdef PointRectDistanceTracker dist_tracker
# dist_tracker = PointRectDistanceTracker(pt, rect, p)
#
# ...
#
# if dist_tracker.min_distance < ...:
#     ...
#
# dist_tracker.push_less_of(node)
# do_something(node.less, dist_tracker)
# dist_tracker.pop()
#
# dist_tracker.push_greater_of(node)
# do_something(node.greater, dist_tracker)
# dist_tracker.pop()

cdef struct RP_stack_item:
    np.intp_t split_dim
    double min_along_dim, max_along_dim
    np.float64_t min_distance, max_distance

cdef class PointRectDistanceTracker(object):
    cdef Rectangle rect
    cdef np.float64_t *pt
    cdef np.float64_t p, epsfac, upper_bound
    cdef np.float64_t min_distance, max_distance
    cdef readonly np.ndarray cperiod

    cdef np.intp_t stack_size, stack_max_size
    cdef RP_stack_item *stack

    # Stack handling
    cdef int _init_stack(self) except -1:
        cdef void *tmp
        self.stack_max_size = 10
        tmp = stdlib.malloc(sizeof(RP_stack_item) *
                            self.stack_max_size)
        if tmp == NULL:
            raise MemoryError
        self.stack = <RP_stack_item*> tmp
        self.stack_size = 0
        return 0

    cdef int _resize_stack(self, np.intp_t new_max_size) except -1:
        cdef void *tmp
        self.stack_max_size = new_max_size
        tmp = stdlib.realloc(<RP_stack_item*> self.stack,
                              new_max_size * sizeof(RP_stack_item))
        if tmp == NULL:
            raise MemoryError
        self.stack = <RP_stack_item*> tmp
        return 0
    
    cdef int _free_stack(self) except -1:
        if self.stack != <RP_stack_item*> NULL:
            stdlib.free(self.stack)
        return 0

    cdef init(self, np.float64_t *pt, Rectangle rect,
              np.float64_t p, np.float64_t eps, np.float64_t upper_bound,
              object period=None):
        
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*rect.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        self.cperiod=cperiod

        self.pt = pt
        self.rect = rect
        self.p = p
        
        # internally we represent all distances as distance ** p
        if p != infinity and upper_bound != infinity:
            self.upper_bound = upper_bound ** p
        else:
            self.upper_bound = upper_bound

        # fiddle approximation factor
        if eps == 0:
            self.epsfac = 1
        elif p == infinity:
            self.epsfac = 1 / (1 + eps)
        else:
            self.epsfac = 1 / (1 + eps) ** p

        self._init_stack()

        # Compute initial min and max distances
        if self.p == infinity:
            self.min_distance = min_dist_point_rect_p_inf(pt, rect)
            self.max_distance = max_dist_point_rect_p_inf(pt, rect)
        else:
            self.min_distance = 0.
            self.max_distance = 0.
            for i in range(rect.m):
                #self.min_distance += min_dist_point_interval_p(pt, rect, i, p)
                #self.max_distance += max_dist_point_interval_p(pt, rect, i, p)
                self.min_distance += min_dist_point_interval_p_periodic(pt, rect, i, p, <np.float64_t*>cperiod.data)
                self.max_distance += max_dist_point_interval_p_periodic(pt, rect, i, p, <np.float64_t*>cperiod.data)

    def __dealloc__(self):
        self._free_stack()

    cdef int push(self, np.intp_t direction,
                  np.intp_t split_dim,
                  np.float64_t split_val) except -1:

        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        cperiod = self.cperiod
        
        # Push onto stack
        if self.stack_size == self.stack_max_size:
            self._resize_stack(self.stack_max_size * 2)
            
        cdef RP_stack_item *item = &self.stack[self.stack_size]
        self.stack_size += 1
        
        item.split_dim = split_dim
        item.min_distance = self.min_distance
        item.max_distance = self.max_distance
        item.min_along_dim = self.rect.mins[split_dim]
        item.max_along_dim = self.rect.maxes[split_dim]
            
        if self.p != infinity:
            #self.min_distance -= min_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)
            #self.max_distance -= max_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)
            self.min_distance -= min_dist_point_interval_p_periodic(self.pt, self.rect, split_dim, self.p, <np.float64_t*>cperiod.data)
            self.max_distance -= max_dist_point_interval_p_periodic(self.pt, self.rect, split_dim, self.p, <np.float64_t*>cperiod.data)

        if direction == LESS:
            self.rect.maxes[split_dim] = split_val
        else:
            self.rect.mins[split_dim] = split_val

        if self.p != infinity:
            #self.min_distance += min_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)
            #self.max_distance += max_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)
            self.min_distance += min_dist_point_interval_p_periodic(self.pt, self.rect, split_dim, self.p, <np.float64_t*>cperiod.data)
            self.max_distance += max_dist_point_interval_p_periodic(self.pt, self.rect, split_dim, self.p, <np.float64_t*>cperiod.data)
        else:
            self.min_distance = min_dist_point_rect_p_inf(self.pt, self.rect)
            self.max_distance = max_dist_point_rect_p_inf(self.pt, self.rect)
            
        return 0

    
    cdef inline int push_less_of(self, innernode* node) except -1:
        return self.push(LESS, node.split_dim, node.split)

    
    cdef inline int push_greater_of(self, innernode* node) except -1:
        return self.push(GREATER, node.split_dim, node.split)

    
    cdef inline int pop(self) except -1:
        self.stack_size -= 1
        assert self.stack_size >= 0
        
        cdef RP_stack_item* item = &self.stack[self.stack_size]
        self.min_distance = item.min_distance
        self.max_distance = item.max_distance
        self.rect.mins[item.split_dim] = item.min_along_dim
        self.rect.maxes[item.split_dim] = item.max_along_dim
        
        return 0

# Tree structure
# ==============
cdef struct innernode:
    np.intp_t split_dim
    np.intp_t children
    np.float64_t* maxes
    np.float64_t* mins
    np.intp_t start_idx
    np.intp_t end_idx
    np.float64_t split
    innernode* less
    innernode* greater
    
cdef struct leafnode:
    np.intp_t split_dim
    np.intp_t children
    np.float64_t* maxes
    np.float64_t* mins
    np.intp_t start_idx
    np.intp_t end_idx

# this is the standard trick for variable-size arrays:
# malloc sizeof(nodeinfo)+self.m*sizeof(np.float64_t) bytes.

cdef struct nodeinfo:
    innernode* node
    np.float64_t side_distances[0]  # FIXME: Only valid in C99, invalid C++ and C89


# Main class
# ==========
cdef class cKDTree:
    """
    cKDTree(data, int leafsize=10)

    kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point. 

    The algorithm used is described in Maneewongvatana and Mount 1999. 
    The general idea is that the kd-tree is a binary trie, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value. 

    During construction, the axis and splitting point are chosen by the 
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin. 

    The tree can be queried for the r closest neighbors of any given point 
    (optionally returning only those within some maximum distance of the 
    point). It can also be queried, with a substantial gain in efficiency, 
    for the r approximate closest neighbors.

    For large dimensions (20 is already large) do not expect this to run 
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    Parameters
    ----------
    data : array-like, shape (n,m)
        The n data points of dimension mto be indexed. This array is 
        not copied unless this is necessary to produce a contiguous 
        array of doubles, and so modifying this data will result in 
        bogus results.
    leafsize : positive integer
        The number of points at which the algorithm switches over to
        brute-force.

    """

    cdef innernode* tree 
    cdef readonly np.ndarray data
    cdef np.float64_t* raw_data
    cdef readonly np.intp_t n, m
    cdef readonly np.intp_t leafsize
    cdef readonly np.ndarray maxes
    cdef np.float64_t* raw_maxes
    cdef readonly np.ndarray mins
    cdef np.float64_t* raw_mins
    cdef readonly np.ndarray indices
    cdef np.intp_t* raw_indices

    def __init__(cKDTree self, data, np.intp_t leafsize=10):
        self.data = np.ascontiguousarray(data,dtype=np.float64)
        self.n, self.m = np.shape(self.data)
        self.leafsize = leafsize
        if self.leafsize<1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.ascontiguousarray(np.amax(self.data,axis=0), dtype=np.float64)
        self.mins = np.ascontiguousarray(np.amin(self.data,axis=0), dtype=np.float64)
        self.indices = np.ascontiguousarray(np.arange(self.n,dtype=np.intp))

        self.raw_data    = <np.float64_t*>np.PyArray_DATA(self.data)
        self.raw_maxes   = <np.float64_t*>np.PyArray_DATA(self.maxes)
        self.raw_mins    = <np.float64_t*>np.PyArray_DATA(self.mins)
        self.raw_indices = <np.intp_t*>np.PyArray_DATA(self.indices)

        self.tree = self.__build(0, self.n, self.raw_maxes, self.raw_mins)

    cdef innernode* __build(cKDTree self, np.intp_t start_idx, np.intp_t end_idx,
                            np.float64_t* maxes, np.float64_t* mins) except? <innernode*> NULL:
        cdef leafnode* n
        cdef innernode* ni
        cdef np.intp_t i, j, t, p, q, d
        cdef np.float64_t size, split, minval, maxval
        cdef np.float64_t*mids
        if end_idx-start_idx<=self.leafsize:
            n = <leafnode*>stdlib.malloc(sizeof(leafnode))
            # Skory
            n.maxes = <np.float64_t*>stdlib.malloc(sizeof(np.float64_t)*self.m)
            n.mins =  <np.float64_t*>stdlib.malloc(sizeof(np.float64_t)*self.m)
            for i in range(self.m):
                n.maxes[i] = maxes[i]
                n.mins[i]  = mins[i]
            if n == <leafnode*> NULL: 
                raise MemoryError
            n.split_dim = -1
            n.children = end_idx - start_idx
            n.start_idx = start_idx
            n.end_idx = end_idx
            return <innernode*>n
        else:
            d = 0 
            size = 0
            for i in range(self.m):
                if maxes[i]-mins[i] > size:
                    d = i
                    size =  maxes[i]-mins[i]
            maxval = maxes[d]
            minval = mins[d]
            if maxval==minval:
                print('may be some duplicate points...')
                n = <leafnode*>stdlib.malloc(sizeof(leafnode))
                if n == <leafnode*> NULL: 
                    raise MemoryError
                n.maxes = <np.float64_t*>stdlib.malloc(sizeof(np.float64_t)*self.m)
                n.mins = <np.float64_t*>stdlib.malloc(sizeof(np.float64_t)*self.m)
                for i in range(self.m):
                    n.mins[i] = mins[i]
                    n.maxes[i] = maxes[i]
                n.split_dim = -1
                n.children = end_idx - start_idx
                n.start_idx = start_idx
                n.end_idx = end_idx
                return <innernode*>n

            split = (maxval+minval)/2

            p = start_idx
            q = end_idx-1
            while p<=q:
                if self.raw_data[self.raw_indices[p]*self.m+d]<split:
                    p+=1
                elif self.raw_data[self.raw_indices[q]*self.m+d]>=split:
                    q-=1
                else:
                    t = self.raw_indices[p]
                    self.raw_indices[p] = self.raw_indices[q]
                    self.raw_indices[q] = t
                    p+=1
                    q-=1

            # slide midpoint if necessary
            if p==start_idx:
                # no points less than split
                j = start_idx
                split = self.raw_data[self.raw_indices[j]*self.m+d]
                for i in range(start_idx+1, end_idx):
                    if self.raw_data[self.raw_indices[i]*self.m+d]<split:
                        j = i
                        split = self.raw_data[self.raw_indices[j]*self.m+d]
                t = self.raw_indices[start_idx]
                self.raw_indices[start_idx] = self.raw_indices[j]
                self.raw_indices[j] = t
                p = start_idx+1
                q = start_idx
            elif p==end_idx:
                # no points greater than split
                j = end_idx-1
                split = self.raw_data[self.raw_indices[j]*self.m+d]
                for i in range(start_idx, end_idx-1):
                    if self.raw_data[self.raw_indices[i]*self.m+d]>split:
                        j = i
                        split = self.raw_data[self.raw_indices[j]*self.m+d]
                t = self.raw_indices[end_idx-1]
                self.raw_indices[end_idx-1] = self.raw_indices[j]
                self.raw_indices[j] = t
                p = end_idx-1
                q = end_idx-2

            # construct new node representation
            ni = <innernode*>stdlib.malloc(sizeof(innernode))
            if ni ==  <innernode*> NULL:
                raise MemoryError

            try:
                mids = <np.float64_t*>stdlib.malloc(sizeof(np.float64_t)*self.m)
                if mids == <np.float64_t*> NULL:
                    raise MemoryError
                        
                for i in range(self.m):
                    mids[i] = maxes[i]
                mids[d] = split
                ni.less = self.__build(start_idx,p,mids,mins)

                for i in range(self.m):
                    mids[i] = mins[i]
                mids[d] = split
                ni.greater = self.__build(p,end_idx,maxes,mids)

                ni.children = ni.less.children + ni.greater.children
            
            except:
                # free ni if it cannot be returned
                if ni !=  <innernode*> NULL:
                    stdlib.free(mids)
                if mids != <np.float64_t*> NULL:
                    stdlib.free(mids)
                raise
            else:
                if mids != <np.float64_t*> NULL:
                    stdlib.free(mids)

            ni.split_dim = d
            ni.split = split
            # Skory
            ni.maxes = <np.float64_t*>stdlib.malloc(sizeof(np.float64_t)*self.m)
            ni.mins = <np.float64_t*>stdlib.malloc(sizeof(np.float64_t)*self.m)
            for i in range(self.m):
                ni.maxes[i] = maxes[i]
                ni.mins[i] = mins[i]
            
            ni.start_idx = start_idx
            ni.end_idx = end_idx
                
            return ni
                    
    cdef __free_tree(cKDTree self, innernode* node):
        if node.split_dim!=-1:
            self.__free_tree(node.less)
            self.__free_tree(node.greater)
        stdlib.free(node.maxes) # Skory
        stdlib.free(node.mins)
        stdlib.free(node)

    def __dealloc__(cKDTree self):
        if <np.intp_t>(self.tree) == 0:
            # should happen only if __init__ was never called
            return
        self.__free_tree(self.tree)

    # -----
    # query
    # -----

    cdef int __query(cKDTree self, 
                     np.float64_t*result_distances, 
                     np.intp_t*result_indices, 
                     np.float64_t*x, 
                     np.intp_t k, 
                     np.float64_t eps, 
                     np.float64_t p, 
                     np.float64_t distance_upper_bound,
                     np.float64_t*period) except -1:

        cdef heap q
        cdef heap neighbors

        cdef np.intp_t i, j
        cdef np.float64_t t
        cdef nodeinfo* inf
        cdef nodeinfo* inf2
        cdef np.float64_t d
        cdef np.float64_t m_left, m_right, m
        cdef np.float64_t epsfac
        cdef np.float64_t min_distance
        cdef np.float64_t far_min_distance
        cdef heapitem it, it2, neighbor
        cdef leafnode* node
        cdef innernode* inode
        cdef innernode* near
        cdef innernode* far
        cdef np.float64_t* side_distances

        # priority queue for chasing nodes
        # entries are:
        #  minimum distance between the cell and the target
        #  distances between the nearest side of the cell and the target
        #  the head node of the cell
        q = heap(12)

        # priority queue for the nearest neighbors
        # furthest known neighbor first
        # entries are (-distance**p, i)
        neighbors = heap(k)

        inf = inf2 = <nodeinfo*> NULL    

        try:
            # set up first nodeinfo
            inf = <nodeinfo*>stdlib.malloc(sizeof(nodeinfo)+self.m*sizeof(np.float64_t))
            if inf == <nodeinfo*> NULL:
                raise MemoryError
            inf.node = self.tree
            for i in range(self.m):
                inf.side_distances[i] = 0
                t = x[i]-self.raw_maxes[i]
                if t>inf.side_distances[i]:
                    inf.side_distances[i] = t
                else:
                    t = self.raw_mins[i]-x[i]
                    if t>inf.side_distances[i]:
                        inf.side_distances[i] = t
                if p!=1 and p!=infinity:
                    inf.side_distances[i]=inf.side_distances[i]**p

            # compute first distance
            min_distance = 0.
            for i in range(self.m):
                if p==infinity:
                    min_distance = dmax(min_distance,inf.side_distances[i])
                else:
                    min_distance += inf.side_distances[i]

            # fiddle approximation factor
            if eps==0:
                epsfac=1
            elif p==infinity:
                epsfac = 1/(1+eps)
            else:
                epsfac = 1/(1+eps)**p

            # internally we represent all distances as distance**p
            if p!=infinity and distance_upper_bound!=infinity:
                distance_upper_bound = distance_upper_bound**p

            while True:
                print('top', inf.node.split_dim)
                print(inf.node.mins[0],inf.node.mins[1],inf.node.maxes[0],inf.node.maxes[1])
                if inf.node.split_dim==-1:
                    node = <leafnode*>inf.node
                    # brute-force
                    for i in range(node.start_idx,node.end_idx):
                        #d = _distance_p(
                        #        self.raw_data+self.raw_indices[i]*self.m,
                        #        x,p,self.m,distance_upper_bound)
                        d = _distance_p_periodic(
                                self.raw_data+self.raw_indices[i]*self.m,
                                x,p,self.m,distance_upper_bound, period)   
                        if d<distance_upper_bound:
                            # replace furthest neighbor
                            if neighbors.n==k:
                                neighbors.remove()
                            neighbor.priority = -d
                            neighbor.contents.intdata = self.raw_indices[i]
                            neighbors.push(neighbor)

                            # adjust upper bound for efficiency
                            if neighbors.n==k:
                                distance_upper_bound = -neighbors.peek().priority
                    
                    # done with this node, get another
                    stdlib.free(inf)
                    inf = <nodeinfo*> NULL

                    if q.n==0:
                        # no more nodes to visit
                        break
                    else:
                        q.pop(&it)
                        inf = <nodeinfo*>it.contents.ptrdata
                        min_distance = it.priority
                else:
                    inode = <innernode*>inf.node

                    # we don't push cells that are too far onto the queue at all,
                    # but since the distance_upper_bound decreases, we might get 
                    # here even if the cell's too far
                    if min_distance>distance_upper_bound*epsfac:

                        # since this is the nearest cell, we're done, bail out
                        stdlib.free(inf)
                        inf = <nodeinfo*> NULL

                        # free all the nodes still on the heap
                        for i in range(q.n):
                            stdlib.free(q.heap[i].contents.ptrdata)
                            q.heap[i].contents.ptrdata = <char*> NULL
                        break

                    # set up children for searching
                    if x[inode.split_dim]<inode.split:
                        near = inode.less
                        far = inode.greater
                    else:
                        near = inode.greater
                        far = inode.less
                    print('bottom', near.split_dim)
                    print(near.mins[0],near.mins[1],near.maxes[0],near.maxes[1])    

                    # near child is at the same distance as the current node
                    # we're going here next, so no point pushing it on the queue
                    # no need to recompute the distance or the side_distances
                    inf.node = near

                    # far child is further by an amount depending only
                    # on the split value; compute its distance and side_distances
                    # and push it on the queue if it's near enough
                    inf2 = <nodeinfo*>stdlib.malloc(sizeof(nodeinfo)+self.m*sizeof(np.float64_t))
                    if inf2 == <nodeinfo*> NULL:
                        raise MemoryError
                    
                    
                    # Periodicity added by S Skory
                    m_left = dmin( dabs(far.mins[inode.split_dim] - x[inode.split_dim]), \
                    period[inode.split_dim] -  dabs(far.mins[inode.split_dim] - x[inode.split_dim]))
                    m_right = dmin( dabs(far.maxes[inode.split_dim] - x[inode.split_dim]), \
                    period[inode.split_dim] -  dabs(far.maxes[inode.split_dim] - x[inode.split_dim]))
                    m = dmin(m_left,m_right)
                    
                    it2.contents.ptrdata = <char*> inf2
                    inf2.node = far
                    # most side distances unchanged
                    for i in range(self.m):
                        inf2.side_distances[i] = inf.side_distances[i]
                        

                    # one side distance changes
                    # we can adjust the minimum distance without recomputing
                    if p == infinity:
                        # we never use side_distances in the l_infinity case
                        # inf2.side_distances[inode.split_dim] = dabs(inode.split-x[inode.split_dim])
                        far_min_distance = dmax(min_distance, dabs(inode.split-x[inode.split_dim]))
                    elif p == 1:
                        inf2.side_distances[inode.split_dim] = dabs(inode.split-x[inode.split_dim])
                        far_min_distance = min_distance - \
                            inf.side_distances[inode.split_dim] + \
                            inf2.side_distances[inode.split_dim]
                    else:
                        #inf2.side_distances[inode.split_dim] = dabs(inode.split - 
                        #                                            x[inode.split_dim])**p
                        inf2.side_distances[inode.split_dim] = m**p
                        #far_min_distance = min_distance - \
                        #    inf.side_distances[inode.split_dim] + \
                        #    inf2.side_distances[inode.split_dim]
                        far_min_distance = m**p

                    it2.priority = far_min_distance


                    # far child might be too far, if so, don't bother pushing it
                    if far_min_distance<=distance_upper_bound*epsfac:
                        q.push(it2)
                    else:
                        stdlib.free(inf2)
                        inf2 = <nodeinfo*> NULL
                        # just in case
                        it2.contents.ptrdata = <char*> NULL

            # fill output arrays with sorted neighbors 
            for i in range(neighbors.n-1,-1,-1):
                neighbors.pop(&neighbor)
                result_indices[i] = neighbor.contents.intdata
                if p==1 or p==infinity:
                    result_distances[i] = -neighbor.priority
                else:
                    result_distances[i] = (-neighbor.priority)**(1./p)
            inf = inf2 = <nodeinfo*> NULL

        finally:
            if inf2 != <nodeinfo*> NULL:
                stdlib.free(inf2)

            if inf != <nodeinfo*> NULL:
                stdlib.free(inf)
        return 0


    @cython.boundscheck(False)
    def query(cKDTree self, object x, np.intp_t k=1, np.float64_t eps=0,
              np.float64_t p=2, np.float64_t distance_upper_bound=infinity,
              object period = None):
        """query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf)
        
        Query the kd-tree for nearest neighbors

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the kth returned value 
            is guaranteed to be no further than (1+eps) times the 
            distance to the real k-th nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use. 
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.

        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors. 
            If x has shape tuple+(self.m,), then d has shape tuple+(k,).
            Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The locations of the neighbors in self.data.
            If `x` has shape tuple+(self.m,), then `i` has shape tuple+(k,).
            Missing neighbors are indicated with self.n.

        """
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        cdef np.ndarray[np.intp_t, ndim=2] ii
        cdef np.ndarray[np.float64_t, ndim=2] dd
        cdef np.ndarray[np.float64_t, ndim=2] xx
        cdef np.intp_t c, n, i, j
        x = np.asarray(x).astype(np.float64)
        if np.shape(x)[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d but has"
                             "shape %s" % (int(self.m), np.shape(x)))
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        if len(x.shape)==1:
            single = True
            x = x[np.newaxis,:]
        else:
            single = False
        retshape = np.shape(x)[:-1]
        n = <np.intp_t> np.prod(retshape)
        xx = np.reshape(x,(n,self.m))
        xx = np.ascontiguousarray(xx,dtype=np.float64)
        dd = np.empty((n,k),dtype=np.float64)
        dd.fill(infinity)
        ii = np.empty((n,k),dtype=np.intp)
        ii.fill(self.n)
        for c in range(n):
            self.__query(&dd[c, 0], &ii[c, 0], &xx[c, 0],
                         k, eps, p, distance_upper_bound, <np.float64_t*>cperiod.data)
        
        if single:
            if k==1:
                if sizeof(long) < sizeof(np.intp_t):
                    # ... e.g. Windows 64
                    if ii[0,0] <= <np.intp_t>LONG_MAX:
                        return dd[0,0], int(ii[0,0])
                    else:
                        return dd[0,0], ii[0,0]
                else:
                    # ... most other platforms
                    return dd[0,0], ii[0,0]
            else:
                return dd[0], ii[0]
        else:
            if sizeof(long) < sizeof(np.intp_t):
                # ... e.g. Windows 64
                for i in range(n):
                    for j in range(k):
                        if ii[i,j] > <np.intp_t>LONG_MAX:
                            # C long overlow, return array of dtype=np.int_p
                            if k==1:
                                return np.reshape(dd[...,0],retshape), np.reshape(ii[...,0],retshape)
                            else:
                                return np.reshape(dd,retshape+(k,)), np.reshape(ii,retshape+(k,))

                # no C long overlow, return array of dtype=int
                if k==1:
                    return np.reshape(dd[...,0],retshape), np.reshape(ii[...,0],retshape).astype(int)
                else:
                    return np.reshape(dd,retshape+(k,)), np.reshape(ii,retshape+(k,)).astype(int)     

            else:
                # ... most other platforms
                if k==1:
                    return np.reshape(dd[...,0],retshape), np.reshape(ii[...,0],retshape)
                else:
                    return np.reshape(dd,retshape+(k,)), np.reshape(ii,retshape+(k,))

    # ----------------
    # query_ball_point
    # ----------------
    cdef int __query_ball_point_traverse_no_checking(cKDTree self,
                                                     list results,
                                                     innernode* node,
                                                     np.float64_t*period) except -1:
        cdef leafnode* lnode
        cdef np.intp_t i

        if node.split_dim == -1:  # leaf node
            lnode = <leafnode*> node
            for i in range(lnode.start_idx, lnode.end_idx):
                list_append(results, self.raw_indices[i])
        else:
            self.__query_ball_point_traverse_no_checking(results, node.less, period)
            self.__query_ball_point_traverse_no_checking(results, node.greater, period)

        return 0


    @cython.cdivision(True)
    cdef int __query_ball_point_traverse_checking(cKDTree self,
                                                  list results,
                                                  innernode* node,
                                                  PointRectDistanceTracker tracker,
                                                  np.float64_t*period) except -1:
        cdef leafnode* lnode
        cdef np.float64_t d
        cdef np.intp_t i

        if tracker.min_distance > tracker.upper_bound * tracker.epsfac:
            return 0
        elif tracker.max_distance < tracker.upper_bound / tracker.epsfac:
            self.__query_ball_point_traverse_no_checking(results, node, period)
        elif node.split_dim == -1:  # leaf node
            lnode = <leafnode*>node
            # brute-force
            for i in range(lnode.start_idx, lnode.end_idx):
                #d = _distance_p(
                #    self.raw_data + self.raw_indices[i] * self.m,
                #    tracker.pt, tracker.p, self.m, tracker.upper_bound)
                d = _distance_p_periodic(
                    self.raw_data + self.raw_indices[i] * self.m,
                    tracker.pt, tracker.p, self.m, tracker.upper_bound, period)
                if d <= tracker.upper_bound:
                    list_append(results, self.raw_indices[i])
        else:
            tracker.push_less_of(node)
            self.__query_ball_point_traverse_checking(
                results, node.less, tracker, period)
            tracker.pop()
            
            tracker.push_greater_of(node)
            self.__query_ball_point_traverse_checking(
                results, node.greater, tracker, period)
            tracker.pop()
            
        return 0


    cdef list __query_ball_point(cKDTree self,
                                 np.float64_t* x,
                                 np.float64_t r,
                                 np.float64_t p,
                                 np.float64_t eps,
                                 object period=None):
                                 
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)

        tracker = PointRectDistanceTracker()
        tracker.init(x, Rectangle(self.mins, self.maxes),
                     p, eps, r, period)
        
        results = []
        self.__query_ball_point_traverse_checking(
            results, self.tree, tracker, <np.float64_t*>cperiod.data)
        return results


    def query_ball_point(cKDTree self, object x, np.float64_t r,
                         np.float64_t p=2., np.float64_t eps=0,
                         object period = None):
        """query_ball_point(self, x, r, p, eps)
        
        Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a cKDTree and using
        query_ball_tree.

        Examples
        --------
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:4, 0:4]
        >>> points = zip(x.ravel(), y.ravel())
        >>> tree = spatial.cKDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [4, 8, 9, 12]

        """
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] xx
        
        x = np.asarray(x).astype(np.float64)
        if x.shape[-1] != self.m:
            raise ValueError("Searching for a %d-dimensional point in a " \
                             "%d-dimensional KDTree" % (int(x.shape[-1]), int(self.m)))
        if len(x.shape) == 1:
            xx = np.ascontiguousarray(x, dtype=np.float64)
            return self.__query_ball_point(&xx[0], r, p, eps, period)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                xx = np.ascontiguousarray(x[c], dtype=np.float64)
                result[c] = self.__query_ball_point(&xx[0], r, p, eps, period)
            return result


    # ----------------
    # query_ball_point_wcounts
    # ----------------
    cdef int __query_ball_point_traverse_no_checking_wcounts(cKDTree self,
                                                             np.float64_t*results,
                                                             innernode* node,
                                                             np.float64_t*period,
                                                             np.float64_t*weights) except -1:
        cdef leafnode* lnode
        cdef np.intp_t i

        if node.split_dim == -1:  # leaf node
            lnode = <leafnode*> node
            for i in range(lnode.start_idx, lnode.end_idx):
                #list_append(results, self.raw_indices[i])
                results[0] += weights[self.raw_indices[i]]
        else:
            self.__query_ball_point_traverse_no_checking_wcounts(results, node.less, period, weights)
            self.__query_ball_point_traverse_no_checking_wcounts(results, node.greater, period, weights)

        return 0


    @cython.cdivision(True)
    cdef int __query_ball_point_traverse_checking_wcounts(cKDTree self,
                                                          np.float64_t*results,
                                                          innernode* node,
                                                          PointRectDistanceTracker tracker,
                                                          np.float64_t*period,
                                                          np.float64_t*weights) except -1:
        cdef leafnode* lnode
        cdef np.float64_t d
        cdef np.intp_t i

        if tracker.min_distance > tracker.upper_bound * tracker.epsfac:
            return 0
        elif tracker.max_distance < tracker.upper_bound / tracker.epsfac:
            self.__query_ball_point_traverse_no_checking_wcounts(results, node, period, weights)
        elif node.split_dim == -1:  # leaf node
            lnode = <leafnode*>node
            # brute-force
            for i in range(lnode.start_idx, lnode.end_idx):
                d = _distance_p_periodic(
                    self.raw_data + self.raw_indices[i] * self.m,
                    tracker.pt, tracker.p, self.m, tracker.upper_bound, period)
                if d <= tracker.upper_bound:
                    #list_append(results, self.raw_indices[i])
                    results[0] += weights[self.raw_indices[i]]
        else:
            tracker.push_less_of(node)
            self.__query_ball_point_traverse_checking_wcounts(
                results, node.less, tracker, period, weights)
            tracker.pop()
            
            tracker.push_greater_of(node)
            self.__query_ball_point_traverse_checking_wcounts(
                results, node.greater, tracker, period, weights)
            tracker.pop()
            
        return 0


    cdef double __query_ball_point_wcounts(cKDTree self,
                                         np.float64_t* x,
                                         np.float64_t r,
                                         np.float64_t p,
                                         np.float64_t eps,
                                         object period=None,
                                         object weights=None):
                                 
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        #process the oweights parameter
        cdef np.ndarray[np.float64_t, ndim=1] cweights
        if weights is None:
            weights = np.array([1.0]*self.n, dtype=np.float64)
        else:
            weights = np.asarray(weights).astype("float64")
        cweights = np.ascontiguousarray(weights) #copy of weights

        tracker = PointRectDistanceTracker()
        tracker.init(x, Rectangle(self.mins, self.maxes),
                     p, eps, r, period)
        
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] results
        results = np.zeros((1,), dtype=np.float64)
        self.__query_ball_point_traverse_checking_wcounts(
            <np.float64_t*>results.data, self.tree, tracker, <np.float64_t*>cperiod.data, <np.float64_t*>cweights.data)
        return results[0]


    def query_ball_point_wcounts(cKDTree self, object x, np.float64_t r,
                                 np.float64_t p=2., np.float64_t eps=0,
                                 object period = None, object weights = None):
        """query_ball_point_wcounts(self, x, r, p, eps)
        
        Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a cKDTree and using
        query_ball_tree.

        Examples
        --------
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:4, 0:4]
        >>> points = zip(x.ravel(), y.ravel())
        >>> tree = spatial.cKDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [4, 8, 9, 12]

        """
        
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] xx
        
        x = np.asarray(x).astype(np.float64)
        if x.shape[-1] != self.m:
            raise ValueError("Searching for a %d-dimensional point in a " \
                             "%d-dimensional KDTree" % (int(x.shape[-1]), int(self.m)))
        if len(x.shape) == 1:
            xx = np.ascontiguousarray(x, dtype=np.float64)
            return self.__query_ball_point_wcounts(&xx[0], r, p, eps, period, weights)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                xx = np.ascontiguousarray(x[c], dtype=np.float64)
                result[c] = self.__query_ball_point_wcounts(&xx[0], r, p, eps, period, weights)
            return result



    # ---------------
    # query_ball_tree
    # ---------------
    cdef int __query_ball_tree_traverse_no_checking(cKDTree self,
                                                    cKDTree other,
                                                    list results,
                                                    innernode* node1,
                                                    innernode* node2,
                                                    np.float64_t*period) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.intp_t i, j
        
        if node1.split_dim == -1:  # leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # leaf node
                lnode2 = <leafnode*>node2
                
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    results_i = results[self.raw_indices[i]]
                    for j in range(lnode2.start_idx, lnode2.end_idx):
                        list_append(results_i, other.raw_indices[j])
            else:
                
                self.__query_ball_tree_traverse_no_checking(other, results, node1, node2.less, period)
                self.__query_ball_tree_traverse_no_checking(other, results, node1, node2.greater, period)
        else:
            
            self.__query_ball_tree_traverse_no_checking(other, results, node1.less, node2, period)
            self.__query_ball_tree_traverse_no_checking(other, results, node1.greater, node2, period)

        return 0


    @cython.cdivision(True)
    cdef int __query_ball_tree_traverse_checking(cKDTree self,
                                                 cKDTree other,
                                                 list results,
                                                 innernode* node1,
                                                 innernode* node2,
                                                 RectRectDistanceTracker tracker,
                                                 np.float64_t*period) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.float64_t d
        cdef np.intp_t i, j

        if tracker.min_distance > tracker.upper_bound * tracker.epsfac:
            return 0
        elif tracker.max_distance < tracker.upper_bound / tracker.epsfac:
            self.__query_ball_tree_traverse_no_checking(other, results, node1, node2, period)
        elif node1.split_dim == -1:  # 1 is leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # 1 & 2 are leaves
                lnode2 = <leafnode*>node2
                
                # brute-force
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    results_i = results[self.raw_indices[i]]
                    for j in range(lnode2.start_idx, lnode2.end_idx):
                        #d = _distance_p(
                        #    self.raw_data + self.raw_indices[i] * self.m,
                        #    other.raw_data + other.raw_indices[j] * other.m,
                        #    tracker.p, self.m, tracker.upper_bound)
                        d = _distance_p_periodic(
                            self.raw_data + self.raw_indices[i] * self.m,
                            other.raw_data + other.raw_indices[j] * other.m,
                            tracker.p, self.m, tracker.upper_bound, period)
                        if d <= tracker.upper_bound:
                            list_append(results_i, other.raw_indices[j])
                            
            else:  # 1 is a leaf node, 2 is inner node

                tracker.push_less_of(2, node2)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1, node2.less, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1, node2.greater, tracker, period)
                tracker.pop()
            
                
        else:  # 1 is an inner node
            if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                tracker.push_less_of(1, node1)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1.less, node2, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(1, node1)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1.greater, node2, tracker, period)
                tracker.pop()
                
            else: # 1 & 2 are inner nodes
                
                tracker.push_less_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1.less, node2.less, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1.less, node2.greater, tracker, period)
                tracker.pop()
                tracker.pop()

                
                tracker.push_greater_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1.greater, node2.less, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_traverse_checking(
                    other, results, node1.greater, node2.greater, tracker, period)
                tracker.pop()
                tracker.pop()
            
        return 0
            

    def query_ball_tree(cKDTree self, cKDTree other,
                        np.float64_t r, np.float64_t p=2., np.float64_t eps=0,
                        object period = None):
        """query_ball_tree(self, other, r, p, eps)

        Find all pairs of points whose distance is at most r

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.

        Returns
        -------
        results : list of lists
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        """
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to query_ball_tree have different dimensionality")

        # Track node-to-node min/max distances
        tracker = RectRectDistanceTracker(
            Rectangle(self.mins, self.maxes),
            Rectangle(other.mins, other.maxes),
            p, eps, r, period)
        
        results = [[] for i in range(self.n)]
        self.__query_ball_tree_traverse_checking(
            other, results, self.tree, other.tree, tracker,  <np.float64_t*>cperiod.data)

        return results


    # ---------------
    # query_ball_tree_wcounts
    # ---------------
    cdef int __query_ball_tree_wcounts_traverse_no_checking(cKDTree self,
                                                            cKDTree other,
                                                            np.float64_t* results,
                                                            innernode* node1,
                                                            innernode* node2,
                                                            np.float64_t*period,
                                                            np.float64_t*sweights,
                                                            np.float64_t*oweights,
                                                            Function w,
                                                            np.float64_t*saux,
                                                            np.float64_t*oaux) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.intp_t i, j
        
        if node1.split_dim == -1:  # leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # leaf node
                lnode2 = <leafnode*>node2
                
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    #results_i = results[self.raw_indices[i]]
                    for j in range(lnode2.start_idx, lnode2.end_idx):
                        #list_append(results_i, other.raw_indices[j])
                        results[self.raw_indices[i]] += w.evaluate(sweights[self.raw_indices[i]],oweights[other.raw_indices[j]],saux[self.raw_indices[i]],oaux[other.raw_indices[j]])
            else:
                
                self.__query_ball_tree_wcounts_traverse_no_checking(other, results, node1, node2.less, period, sweights, oweights, w, saux, oaux)
                self.__query_ball_tree_wcounts_traverse_no_checking(other, results, node1, node2.greater, period, sweights, oweights, w, saux, oaux)
        else:
            
            self.__query_ball_tree_wcounts_traverse_no_checking(other, results, node1.less, node2, period, sweights, oweights, w, saux, oaux)
            self.__query_ball_tree_wcounts_traverse_no_checking(other, results, node1.greater, node2, period, sweights, oweights, w, saux, oaux)

        return 0


    @cython.cdivision(True)
    cdef int __query_ball_tree_wcounts_traverse_checking(cKDTree self,
                                                         cKDTree other,
                                                         np.float64_t* results,
                                                         innernode* node1,
                                                         innernode* node2,
                                                         RectRectDistanceTracker tracker,
                                                         np.float64_t*period,
                                                         np.float64_t*sweights,
                                                         np.float64_t*oweights,
                                                         Function w,
                                                         np.float64_t*saux,
                                                         np.float64_t*oaux) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.float64_t d
        cdef np.intp_t i, j

        if tracker.min_distance > tracker.upper_bound * tracker.epsfac:
            return 0
        elif tracker.max_distance < tracker.upper_bound / tracker.epsfac:
            self.__query_ball_tree_wcounts_traverse_no_checking(other, results, node1, node2, period, sweights, oweights, w, saux, oaux)
        elif node1.split_dim == -1:  # 1 is leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # 1 & 2 are leaves
                lnode2 = <leafnode*>node2
                
                # brute-force
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    #results_i = results[self.raw_indices[i]]
                    for j in range(lnode2.start_idx, lnode2.end_idx):
                        #d = _distance_p(
                        #    self.raw_data + self.raw_indices[i] * self.m,
                        #    other.raw_data + other.raw_indices[j] * other.m,
                        #    tracker.p, self.m, tracker.upper_bound)
                        d = _distance_p_periodic(
                            self.raw_data + self.raw_indices[i] * self.m,
                            other.raw_data + other.raw_indices[j] * other.m,
                            tracker.p, self.m, tracker.upper_bound, period)
                        if d <= tracker.upper_bound:
                            #list_append(results_i, other.raw_indices[j])
                            results[self.raw_indices[i]] += w.evaluate(sweights[self.raw_indices[i]],oweights[other.raw_indices[j]],saux[self.raw_indices[i]],oaux[other.raw_indices[j]])
                            
            else:  # 1 is a leaf node, 2 is inner node

                tracker.push_less_of(2, node2)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1, node2.less, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1, node2.greater, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
            
                
        else:  # 1 is an inner node
            if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                tracker.push_less_of(1, node1)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1.less, node2, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
                    
                tracker.push_greater_of(1, node1)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1.greater, node2, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
                
            else: # 1 & 2 are inner nodes
                
                tracker.push_less_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1.less, node2.less, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1.less, node2.greater, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
                tracker.pop()

                
                tracker.push_greater_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1.greater, node2.less, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_wcounts_traverse_checking(
                    other, results, node1.greater, node2.greater, tracker, period, sweights, oweights, w, saux, oaux)
                tracker.pop()
                tracker.pop()
            
        return 0
            

    def query_ball_tree_wcounts(cKDTree self, cKDTree other,
                        np.float64_t r, np.float64_t p=2., np.float64_t eps=0,
                        object period = None,
                        object sweights = None, object oweights = None,
                        Function w=None,
                        object saux = None, object oaux = None):
        """query_ball_tree_wcounts_counts(self, other, r, p, eps, period, weights)

        Find all weighted pair counts of points whose distance is at most r

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.
        weights : array_like, dimension other.n
            A vector indicating the weight attached to each point in other.

        Returns
        -------
        results : list of floats
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            float of the weighted count of its neighbors in ``other.data``.

        """
        
        #process count function
        if w is None:
            w = fmultiply()
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        #process the oweights parameter
        cdef np.ndarray[np.float64_t, ndim=1] coweights
        if oweights is None:
            oweights = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oweights).astype("float64")
        coweights = np.ascontiguousarray(oweights) #copy of oweights
    
        #process the sweights parameter
        cdef np.ndarray[np.float64_t, ndim=1] csweights
        if sweights is None:
            sweights = np.array([1.0]*self.n, dtype=np.float64)
        else:
            sweights = np.asarray(sweights).astype("float64")
        csweights = np.ascontiguousarray(sweights) #copy of sweights
        
        #process the self aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] csaux #copy of self weights
        if saux is None:
            saux = np.array([1.0]*self.n, dtype=np.float64)
        else:
            saux = np.asarray(saux).astype("float64")
        csaux = np.ascontiguousarray(saux)
        
        #process the other aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] coaux #copy of other weights
        if oaux is None:
            oaux = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oaux).astype("float64")
        coaux = np.ascontiguousarray(oaux)

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to query_ball_tree_wcounts have different dimensionality")

        # Track node-to-node min/max distances
        tracker = RectRectDistanceTracker(
            Rectangle(self.mins, self.maxes),
            Rectangle(other.mins, other.maxes),
            p, eps, r, period)
        
        #results = [[] for i in range(self.n)]
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] results
        results = np.zeros((self.n,), dtype=np.float64)
        #self.__query_ball_tree_wcounts_traverse_checking(other, results, self.tree,
        #                                                 other.tree, tracker,
        #                                                 <np.float64_t*>cperiod.data,
        #                                                 <np.float64_t*>coweights.data)
        self.__query_ball_tree_wcounts_traverse_checking(other, &results[0], self.tree,
                                                         other.tree, tracker,
                                                         <np.float64_t*>cperiod.data,
                                                         <np.float64_t*>csweights.data,
                                                         <np.float64_t*>coweights.data,
                                                         w,
                                                         <np.float64_t*>csaux.data,
                                                         <np.float64_t*>coaux.data)

        return results


    # -----------
    # query_pairs
    # -----------
    cdef int __query_pairs_traverse_no_checking(cKDTree self,
                                                set results,
                                                innernode* node1,
                                                innernode* node2,
                                                np.float64_t*period) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.intp_t i, j, min_j
        
        if node1.split_dim == -1:  # leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # leaf node
                lnode2 = <leafnode*>node2

                for i in range(lnode1.start_idx, lnode1.end_idx):
                    # Special care here to avoid duplicate pairs
                    if node1 == node2:
                        min_j = i + 1
                    else:
                        min_j = lnode2.start_idx
                        
                    for j in range(min_j, lnode2.end_idx):
                        set_add_ordered_pair(results,
                                             self.raw_indices[i],
                                             self.raw_indices[j])
                            
            else:
                self.__query_pairs_traverse_no_checking(results, node1, node2.less, period)
                self.__query_pairs_traverse_no_checking(results, node1, node2.greater, period)
        else:
            if node1 == node2:
                # Avoid traversing (node1.less, node2.greater) and
                # (node1.greater, node2.less) (it's the same node pair twice
                # over, which is the source of the complication in the
                # original KDTree.query_pairs)
                self.__query_pairs_traverse_no_checking(results, node1.less, node2.less, period)
                self.__query_pairs_traverse_no_checking(results, node1.less, node2.greater, period)
                self.__query_pairs_traverse_no_checking(results, node1.greater, node2.greater, period)
            else:
                self.__query_pairs_traverse_no_checking(results, node1.less, node2, period)
                self.__query_pairs_traverse_no_checking(results, node1.greater, node2, period)

        return 0

    @cython.cdivision(True)
    cdef int __query_pairs_traverse_checking(cKDTree self,
                                             set results,
                                             innernode* node1,
                                             innernode* node2,
                                             RectRectDistanceTracker tracker,
                                             np.float64_t*period) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.float64_t d
        cdef np.intp_t i, j, min_j

        if tracker.min_distance > tracker.upper_bound * tracker.epsfac:
            return 0
        elif tracker.max_distance < tracker.upper_bound / tracker.epsfac:
            self.__query_pairs_traverse_no_checking(results, node1, node2, period)
        elif node1.split_dim == -1:  # 1 is leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # 1 & 2 are leaves
                lnode2 = <leafnode*>node2
                
                # brute-force
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    
                    # Special care here to avoid duplicate pairs
                    if node1 == node2:
                        min_j = i + 1
                    else:
                        min_j = lnode2.start_idx
                        
                    for j in range(min_j, lnode2.end_idx):
                        #d = _distance_p(
                        #    self.raw_data + self.raw_indices[i] * self.m,
                        #    self.raw_data + self.raw_indices[j] * self.m,
                        #    tracker.p, self.m, tracker.upper_bound)
                        d = _distance_p_periodic(
                            self.raw_data + self.raw_indices[i] * self.m,
                            self.raw_data + self.raw_indices[j] * self.m,
                            tracker.p, self.m, tracker.upper_bound, period)
                        if d <= tracker.upper_bound:
                            set_add_ordered_pair(results,
                                                 self.raw_indices[i],
                                                 self.raw_indices[j])
                            
            else:  # 1 is a leaf node, 2 is inner node
                tracker.push_less_of(2, node2)
                self.__query_pairs_traverse_checking(
                    results, node1, node2.less, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_pairs_traverse_checking(
                    results, node1, node2.greater, tracker, period)
                tracker.pop()
                
        else:  # 1 is an inner node
            if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                tracker.push_less_of(1, node1)
                self.__query_pairs_traverse_checking(
                    results, node1.less, node2, tracker, period)
                tracker.pop()
                
                tracker.push_greater_of(1, node1)
                self.__query_pairs_traverse_checking(
                    results, node1.greater, node2, tracker, period)
                tracker.pop()
                
            else: # 1 and 2 are inner nodes
                tracker.push_less_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__query_pairs_traverse_checking(
                    results, node1.less, node2.less, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_pairs_traverse_checking(
                    results, node1.less, node2.greater, tracker, period)
                tracker.pop()
                tracker.pop()
                    
                tracker.push_greater_of(1, node1)
                if node1 != node2:
                    # Avoid traversing (node1.less, node2.greater) and
                    # (node1.greater, node2.less) (it's the same node pair
                    # twice over, which is the source of the complication in
                    # the original KDTree.query_pairs)
                    tracker.push_less_of(2, node2)
                    self.__query_pairs_traverse_checking(
                        results, node1.greater, node2.less, tracker, period)
                    tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_pairs_traverse_checking(
                    results, node1.greater, node2.greater, tracker, period)
                tracker.pop()
                tracker.pop()
                
        return 0
            

    def query_pairs(cKDTree self, np.float64_t r, np.float64_t p=2.,
                    np.float64_t eps=0, object period = None):
        """query_pairs(self, r, p, eps)

        Find all pairs of points whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.

        Returns
        -------
        results : set
            Set of pairs ``(i,j)``, with ``i < j`, for which the corresponding
            positions are close.

        """
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        tracker = RectRectDistanceTracker(
            Rectangle(self.mins, self.maxes),
            Rectangle(self.mins, self.maxes),
            p, eps, r, period)
        
        results = set()
        self.__query_pairs_traverse_checking(
            results, self.tree, self.tree, tracker,  <np.float64_t*>cperiod.data)
        
        return results


    # ---------------
    # count_neighbors
    # ---------------
    cdef int __count_neighbors_traverse(cKDTree self,
                                        cKDTree other,
                                        np.intp_t n_queries,
                                        np.float64_t* r,
                                        np.intp_t * results,
                                        np.intp_t * idx,
                                        innernode* node1,
                                        innernode* node2,
                                        RectRectDistanceTracker tracker,
                                        np.float64_t*period) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef np.float64_t d
        cdef np.intp_t *old_idx
        cdef np.intp_t old_n_queries, l, i, j

        # Speed through pairs of nodes all of whose children are close
        # and see if any work remains to be done
        old_idx = idx
        cdef np.ndarray[np.intp_t, ndim=1] inner_idx
        inner_idx = np.empty((n_queries,), dtype=np.intp)
        idx = &inner_idx[0]

        #if node1.children * node2.children < 100.0:
        #    return 0

        old_n_queries = n_queries
        n_queries = 0
        for i in range(old_n_queries):
            if tracker.max_distance < r[old_idx[i]] / tracker.epsfac:
                results[old_idx[i]] += node1.children * node2.children
            elif tracker.min_distance <= r[old_idx[i]] * tracker.epsfac:
                idx[n_queries] = old_idx[i]
                n_queries += 1
                    

        if n_queries > 0:
            # OK, need to probe a bit deeper
            if node1.split_dim == -1:  # 1 is leaf node
                lnode1 = <leafnode*>node1
                if node2.split_dim == -1:  # 1 & 2 are leaves
                    lnode2 = <leafnode*>node2
                    
                    # brute-force
                    for i in range(lnode1.start_idx, lnode1.end_idx):
                        for j in range(lnode2.start_idx, lnode2.end_idx):
                            #d = _distance_p(
                            #    self.raw_data + self.raw_indices[i] * self.m,
                            #    other.raw_data + other.raw_indices[j] * other.m,
                            #    tracker.p, self.m, tracker.max_distance)
                            d = _distance_p_periodic(
                                self.raw_data + self.raw_indices[i] * self.m,
                                other.raw_data + other.raw_indices[j] * other.m,
                                tracker.p, self.m, tracker.max_distance, period)
                            # I think it's usually cheaper to test d against all r's
                            # than to generate a distance array, sort it, then
                            # search for all r's via binary search
                            for l in range(n_queries):
                                if d <= r[idx[l]]:
                                    results[idx[l]] += 1
                                
                else:  # 1 is a leaf node, 2 is inner node
                    tracker.push_less_of(2, node2)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.less, tracker, period)
                    tracker.pop()

                    tracker.push_greater_of(2, node2)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.greater, tracker, period)
                    tracker.pop()
                
            else:  # 1 is an inner node
                if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                    tracker.push_less_of(1, node1)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2, tracker, period)
                    tracker.pop()
                    
                    tracker.push_greater_of(1, node1)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2, tracker, period)
                    tracker.pop()
                    
                else: # 1 and 2 are inner nodes
                    tracker.push_less_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.less, tracker, period)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.greater, tracker, period)
                    tracker.pop()
                    tracker.pop()
                        
                    tracker.push_greater_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.less, tracker, period)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__count_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.greater, tracker, period)
                    tracker.pop()
                    tracker.pop()
                    
        return 0

    @cython.boundscheck(False)
    def count_neighbors(cKDTree self, cKDTree other, object r, np.float64_t p=2.,
                        object period = None, eps=0.0):
        """count_neighbors(self, other, r, p)

        Count how many nearby pairs can be formed.

        Count the number of pairs (x1,x2) can be formed, with x1 drawn
        from self and x2 drawn from `other`, and where
        ``distance(x1, x2, p) <= r``.
        This is the "two-point correlation" described in Gray and Moore 2000,
        "N-body problems in statistical learning", and the code here is based
        on their algorithm.

        Parameters
        ----------
        other : KDTree instance
            The other tree to draw points from.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        result : int or 1-D array of ints
            The number of pairs. Note that this is internally stored in a numpy int,
            and so may overflow if very large (2e9).

        """
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        cdef np.intp_t n_queries, i
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] real_r
        cdef np.ndarray[np.intp_t, ndim=1, mode="c"] results, idx

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to count_neighbors have different dimensionality")

        # Make a copy of r array to ensure it's contiguous and to modify it
        # below
        if np.shape(r) == ():
            real_r = np.array([r], dtype=np.float64)
            n_queries = 1
        elif len(np.shape(r))==1:
            real_r = np.array(r, dtype=np.float64)
            n_queries = r.shape[0]
        else:
            raise ValueError("r must be either a single value or a one-dimensional array of values")

        # Internally, we represent all distances as distance ** p
        if p != infinity:
            for i in range(n_queries):
                if real_r[i] != infinity:
                    real_r[i] = real_r[i] ** p

        # Track node-to-node min/max distances
        tracker = RectRectDistanceTracker(
            Rectangle(self.mins, self.maxes),
            Rectangle(other.mins, other.maxes),
            p, eps, 0.0, period)
        
        # Go!
        results = np.zeros((n_queries,), dtype=np.intp)
        idx = np.arange(n_queries, dtype=np.intp)
        self.__count_neighbors_traverse(other, n_queries,
                                        &real_r[0], &results[0], &idx[0],
                                        self.tree, other.tree,
                                        tracker,  <np.float64_t*>cperiod.data)
        
        if np.shape(r) == ():
            if results[0] <= <np.intp_t> LONG_MAX:
                return int(results[0])
            else:
                return results[0]
        elif len(np.shape(r))==1:
            return results

    # ---------------
    # wcount_neighbors
    # ---------------
    cdef int __wcount_neighbors_traverse(cKDTree self,
                                         cKDTree other,
                                         np.intp_t n_queries,
                                         np.float64_t* r,
                                         np.float64_t* results,
                                         np.intp_t * idx,
                                         innernode* node1,
                                         innernode* node2,
                                         RectRectDistanceTracker tracker,
                                         np.float64_t*period,
                                         np.float64_t*sweights,
                                         np.float64_t*oweights,
                                         Function w,
                                         np.float64_t*saux,
                                         np.float64_t*oaux) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef np.float64_t d
        cdef np.intp_t *old_idx
        cdef np.intp_t old_n_queries, l, i, j
        cdef np.float64_t wsum

        # Speed through pairs of nodes all of whose children are close
        # and see if any work remains to be done
        old_idx = idx
        cdef np.ndarray[np.intp_t, ndim=1] inner_idx
        inner_idx = np.empty((n_queries,), dtype=np.intp)
        idx = &inner_idx[0]

        old_n_queries = n_queries
        n_queries = 0
        for i in range(old_n_queries):
            if tracker.max_distance < r[old_idx[i]]:
                #need to run through all children and sum weights
                wsum = 0.0
                for j in range(node2.start_idx, node2.end_idx): #Fuck yes.
                    for k in range(node1.start_idx, node1.end_idx):
                        wsum += w.evaluate(sweights[self.raw_indices[k]],
                                           oweights[other.raw_indices[j]],
                                           saux[self.raw_indices[k]],
                                           oaux[other.raw_indices[j]])
                results[old_idx[i]] += wsum
            elif tracker.min_distance <= r[old_idx[i]]:
                idx[n_queries] = old_idx[i]
                n_queries += 1

        if n_queries > 0:
            # OK, need to probe a bit deeper
            if node1.split_dim == -1:  # 1 is leaf node
                lnode1 = <leafnode*>node1
                if node2.split_dim == -1:  # 1 & 2 are leaves
                    lnode2 = <leafnode*>node2
                    
                    # brute-force
                    for i in range(lnode1.start_idx, lnode1.end_idx):
                        for j in range(lnode2.start_idx, lnode2.end_idx):
                            d = _distance_p_periodic(
                                self.raw_data + self.raw_indices[i] * self.m,
                                other.raw_data + other.raw_indices[j] * other.m,
                                tracker.p, self.m, tracker.max_distance, period)
                            # I think it's usually cheaper to test d against all r's
                            # than to generate a distance array, sort it, then
                            # search for all r's via binary search
                            for l in range(n_queries):
                                if d <= r[idx[l]]:
                                    results[idx[l]] += w.evaluate(sweights[self.raw_indices[i]],
                                                                  oweights[other.raw_indices[j]],
                                                                  saux[self.raw_indices[i]],
                                                                  oaux[other.raw_indices[j]])
                                
                else:  # 1 is a leaf node, 2 is inner node
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()

                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                
            else:  # 1 is an inner node
                if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                    tracker.push_less_of(1, node1)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    
                    tracker.push_greater_of(1, node1)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    
                else: # 1 and 2 are inner nodes
                    tracker.push_less_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    tracker.pop()
                        
                    tracker.push_greater_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    tracker.pop()
                    
        return 0

    @cython.boundscheck(False)
    def wcount_neighbors(cKDTree self, cKDTree other,
                         object r, np.float64_t p=2.,
                         object period = None, 
                         object sweights = None, object oweights = None, 
                         Function w=None, 
                         object saux=None, object oaux=None):
        """wcount_neighbors(self, other, r, p)

        Weighted count of how many nearby pairs can be formed.

        Count the number of pairs (x1,x2) can be formed, with x1 drawn
        from self and x2 drawn from `other`, and where
        ``distance(x1, x2, p) <= r``.
        This is the "two-point correlation" described in Gray and Moore 2000,
        "N-body problems in statistical learning", and the code here is based
        on their algorithm.

        Parameters
        ----------
        other : KDTree instance
            The other tree to draw points from.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.
        sweights : array_like, dimension self.n
            A vector indicating the weight attached to each point in self.
        oweights : array_like, dimension other.n
            A vector indicating the weight attached to each point in other.
        w: ckdtree.Function object.  Function used in weighting.  None results in w1*w2
            w(self weight, other weight, self aux_data, other aux_data)

        Returns
        -------
        result : float or 1-D array of floats
            The weighted number of pairs. Note that this is internally stored in a numpy float,
            and so may overflow if very large (2e9).

        """
        
        #process count function
        if w is None:
            w = fmultiply()
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        cdef np.intp_t n_queries, i
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] real_r
        cdef np.ndarray[np.intp_t, ndim=1, mode="c"] idx
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] results
        
        #process the self weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] csweights #copy of self weights
        if sweights is None:
            sweights = np.array([1.0]*self.n, dtype=np.float64)
        else:
            sweights = np.asarray(sweights).astype("float64")
        csweights = np.ascontiguousarray(sweights)
        
        #process the other weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] coweights #copy of other weights
        if oweights is None:
            oweights = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oweights).astype("float64")
        coweights = np.ascontiguousarray(oweights)
        
        #process the self aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] csaux #copy of self weights
        if saux is None:
            saux = np.array([1.0]*self.n, dtype=np.float64)
        else:
            saux = np.asarray(saux).astype("float64")
        csaux = np.ascontiguousarray(saux)
        
        #process the other aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] coaux #copy of other weights
        if oaux is None:
            oaux = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oaux).astype("float64")
        coaux = np.ascontiguousarray(oaux)

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to count_neighbors have different dimensionality")

        # Make a copy of r array to ensure it's contiguous and to modify it below
        if np.shape(r) == ():
            real_r = np.array([r], dtype=np.float64)
            n_queries = 1
        elif len(np.shape(r))==1:
            real_r = np.array(r, dtype=np.float64)
            n_queries = r.shape[0]
        else:
            raise ValueError("r must be either a single value or a one-dimensional array of values")

        # Internally, we represent all distances as distance ** p
        if p != infinity:
            for i in range(n_queries):
                if real_r[i] != infinity:
                    real_r[i] = real_r[i] ** p

        # Track node-to-node min/max distances
        tracker = RectRectDistanceTracker(
            Rectangle(self.mins, self.maxes),
            Rectangle(other.mins, other.maxes),
            p, 0.0, 0.0, period)
        
        # Go!
        results = np.zeros((n_queries,), dtype=np.float64)
        idx = np.arange(n_queries, dtype=np.intp)
        self.__wcount_neighbors_traverse(other, n_queries,
                                        &real_r[0], &results[0], &idx[0],
                                        self.tree, other.tree,
                                        tracker,  <np.float64_t*>cperiod.data,
                                        <np.float64_t*>csweights.data,
                                        <np.float64_t*>coweights.data,
                                        w, 
                                        <np.float64_t*>csaux.data,
                                        <np.float64_t*>coaux.data)

        if np.shape(r) == ():
            return results[0]
        elif len(np.shape(r))==1:
            return results


    # ---------------
    # wcount_neighbors_custom
    # ---------------
    cdef int __wcount_neighbors_custom_traverse(cKDTree self,
                                                cKDTree other,
                                                np.intp_t n_queries,
                                                np.float64_t* r,
                                                np.float64_t[:,:] results,
                                                np.intp_t * idx,
                                                innernode* node1,
                                                innernode* node2,
                                                RectRectDistanceTracker tracker,
                                                np.float64_t*period,
                                                np.float64_t*sweights,
                                                np.float64_t*oweights,
                                                Function w,
                                                np.float64_t*saux,
                                                np.float64_t*oaux) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef np.float64_t d
        cdef np.intp_t *old_idx
        cdef np.intp_t old_n_queries, l, i, j
        cdef np.float64_t wsum

        # Speed through pairs of nodes all of whose children are close
        # and see if any work remains to be done
        old_idx = idx
        cdef np.ndarray[np.intp_t, ndim=1] inner_idx
        inner_idx = np.empty((n_queries,), dtype=np.intp)
        idx = &inner_idx[0]

        old_n_queries = n_queries
        n_queries = 0
        for i in range(old_n_queries):
            if tracker.max_distance < r[old_idx[i]]:
                #need to run through all children and sum weights
                for k in range(node1.start_idx, node1.end_idx):
                    wsum = 0.0
                    for j in range(node2.start_idx, node2.end_idx):
                        wsum += w.evaluate(sweights[self.raw_indices[k]],
                                           oweights[other.raw_indices[j]],
                                           saux[self.raw_indices[k]],
                                           oaux[other.raw_indices[j]])
                    results[self.raw_indices[k],old_idx[i]] += wsum
            elif tracker.min_distance <= r[old_idx[i]]:
                idx[n_queries] = old_idx[i]
                n_queries += 1

        if n_queries > 0:
            # OK, need to probe a bit deeper
            if node1.split_dim == -1:  # 1 is leaf node
                lnode1 = <leafnode*>node1
                if node2.split_dim == -1:  # 1 & 2 are leaves
                    lnode2 = <leafnode*>node2
                    
                    # brute-force
                    for i in range(lnode1.start_idx, lnode1.end_idx):
                        for j in range(lnode2.start_idx, lnode2.end_idx):
                            d = _distance_p_periodic(
                                self.raw_data + self.raw_indices[i] * self.m,
                                other.raw_data + other.raw_indices[j] * other.m,
                                tracker.p, self.m, tracker.max_distance, period)
                            # I think it's usually cheaper to test d against all r's
                            # than to generate a distance array, sort it, then
                            # search for all r's via binary search
                            for l in range(n_queries):
                                if d <= r[idx[l]]:
                                    results[self.raw_indices[i],idx[l]] += w.evaluate(
                                                                  sweights[self.raw_indices[i]],
                                                                  oweights[other.raw_indices[j]],
                                                                  saux[self.raw_indices[i]],
                                                                  oaux[other.raw_indices[j]])
                                
                else:  # 1 is a leaf node, 2 is inner node
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.less, tracker, period,
                        sweights, oweights, w, saux, oaux)
                    tracker.pop()

                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                
            else:  # 1 is an inner node
                if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                    tracker.push_less_of(1, node1)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    
                    tracker.push_greater_of(1, node1)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    
                else: # 1 and 2 are inner nodes
                    tracker.push_less_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    tracker.pop()
                        
                    tracker.push_greater_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_custom_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    tracker.pop()
                    
        return 0

    @cython.boundscheck(False)
    def wcount_neighbors_custom(cKDTree self, cKDTree other, 
                                object r, np.float64_t p=2.,
                                object period = None, 
                                object sweights = None, object oweights = None,
                                Function w=None, object saux=None, object oaux=None):
        """wcount_neighbors_custom(self, other, r, p)

        Weighted count of how many nearby pairs can be formed for each point in self.

        Count the number of pairs (x1,x2) can be formed, with x1 drawn
        from self and x2 drawn from `other`, and where
        ``distance(x1, x2, p) <= r``.
        This is the "two-point correlation" described in Gray and Moore 2000,
        "N-body problems in statistical learning", and the code here is based
        on their algorithm.

        Parameters
        ----------
        other : KDTree instance
            The other tree to draw points from.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.
        sweights : array_like, dimension self.n
            A vector indicating the weight attached to each point in self.
        oweights : array_like, dimension other.n
            A vector indicating the weight attached to each point in other.
        w: ckdtree.Function object.  Function used in weighting.  None results in w1*w2
            w(self weight, other weight, self aux_data, other aux_data)

        Returns
        -------
        result : float or 1-D array of floats
            The weighted number of pairs. Note that this is internally stored in a numpy float,
            and so may overflow if very large (2e9).

        """
        
        #process count function
        if w is None:
            w = fmultiply()
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        cdef np.intp_t n_queries, i
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] real_r
        cdef np.ndarray[np.intp_t, ndim=1, mode="c"] idx
        
        #process the self weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] csweights #copy of self weights
        if sweights is None:
            sweights = np.array([1.0]*self.n, dtype=np.float64)
        else:
            sweights = np.asarray(sweights).astype("float64")
        csweights = np.ascontiguousarray(sweights)
        
        #process the other weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] coweights #copy of other weights
        if oweights is None:
            oweights = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oweights).astype("float64")
        coweights = np.ascontiguousarray(oweights)
        
        #process the self aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] csaux #copy of self weights
        if saux is None:
            saux = np.array([1.0]*self.n, dtype=np.float64)
        else:
            saux = np.asarray(saux).astype("float64")
        csaux = np.ascontiguousarray(saux)
        
        #process the other aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] coaux #copy of other weights
        if oaux is None:
            oaux = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oaux).astype("float64")
        coaux = np.ascontiguousarray(oaux)

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to count_neighbors have different dimensionality")

        # Make a copy of r array to ensure it's contiguous and to modify it below
        if np.shape(r) == ():
            real_r = np.array([r], dtype=np.float64)
            n_queries = 1
        elif len(np.shape(r))==1:
            real_r = np.array(r, dtype=np.float64)
            n_queries = r.shape[0]
        else:
            raise ValueError("r must be either a single value or a one-dimensional array of values")

        # Internally, we represent all distances as distance ** p
        if p != infinity:
            for i in range(n_queries):
                if real_r[i] != infinity:
                    real_r[i] = real_r[i] ** p

        # Track node-to-node min/max distances
        tracker = RectRectDistanceTracker(Rectangle(self.mins, self.maxes),
                                          Rectangle(other.mins, other.maxes),
                                          p, 0.0, 0.0, period)
        
        results = np.zeros(self.n*n_queries, dtype=np.float64).reshape((self.n,n_queries))
        cdef np.float64_t [:, :] results_view = results
        idx = np.arange(n_queries, dtype=np.intp)
        self.__wcount_neighbors_custom_traverse(other, n_queries,
                                                &real_r[0], results_view, &idx[0],
                                                self.tree, other.tree,
                                                tracker,  <np.float64_t*>cperiod.data,
                                                <np.float64_t*>csweights.data,
                                                <np.float64_t*>coweights.data,
                                                w, 
                                                <np.float64_t*>csaux.data,
                                                <np.float64_t*>coaux.data)

        if np.shape(r) == ():
            return results[:,0]
        elif len(np.shape(r))==1:
            return results
    
    
    # ---------------
    # wcount_neighbors_custom_2D
    # ---------------
    cdef int __wcount_neighbors_custom_2D_traverse(cKDTree self,
                                                cKDTree other,
                                                np.intp_t n_queries,
                                                np.float64_t* r,
                                                np.float64_t[:,:] results,
                                                np.intp_t * idx,
                                                innernode* node1,
                                                innernode* node2,
                                                RectRectDistanceTracker tracker,
                                                np.float64_t*period,
                                                np.float64_t*sweights,
                                                np.float64_t*oweights,
                                                Function w,
                                                np.float64_t*saux,
                                                np.float64_t*oaux,
                                                np.intp_t wdim) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef np.float64_t d
        cdef np.intp_t *old_idx
        cdef np.intp_t old_n_queries, l, i, j
        cdef np.float64_t wsum

        # Speed through pairs of nodes all of whose children are close
        # and see if any work remains to be done
        old_idx = idx
        cdef np.ndarray[np.intp_t, ndim=1] inner_idx
        inner_idx = np.empty((n_queries,), dtype=np.intp)
        idx = &inner_idx[0]

        old_n_queries = n_queries
        n_queries = 0
        for i in range(old_n_queries):
            if tracker.max_distance < r[old_idx[i]]:
                #need to run through all children and sum weights
                for k in range(node1.start_idx, node1.end_idx):
                    for l in range(0,wdim):
                        wsum = 0.0
                        for j in range(node2.start_idx, node2.end_idx):
                            wsum += w.evaluate(sweights[self.raw_indices[k]],
                                               oweights[other.raw_indices[j]],
                                               l,wdim)
                        results[l,old_idx[i]] += wsum
            elif tracker.min_distance <= r[old_idx[i]]:
                idx[n_queries] = old_idx[i]
                n_queries += 1

        if n_queries > 0:
            # OK, need to probe a bit deeper
            if node1.split_dim == -1:  # 1 is leaf node
                lnode1 = <leafnode*>node1
                if node2.split_dim == -1:  # 1 & 2 are leaves
                    lnode2 = <leafnode*>node2
                    
                    # brute-force
                    for i in range(lnode1.start_idx, lnode1.end_idx):
                        for j in range(lnode2.start_idx, lnode2.end_idx):
                            d = _distance_p_periodic(
                                self.raw_data + self.raw_indices[i] * self.m,
                                other.raw_data + other.raw_indices[j] * other.m,
                                tracker.p, self.m, tracker.max_distance, period)
                            # I think it's usually cheaper to test d against all r's
                            # than to generate a distance array, sort it, then
                            # search for all r's via binary search
                            for l in range(n_queries):
                                if d <= r[idx[l]]:
                                    for k in range(0,wdim):
                                        results[k,idx[l]] += w.evaluate(
                                                                  sweights[self.raw_indices[i]],
                                                                  oweights[other.raw_indices[j]],
                                                                  k,wdim)
                                        
                                
                else:  # 1 is a leaf node, 2 is inner node
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()

                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()
                
            else:  # 1 is an inner node
                if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                    tracker.push_less_of(1, node1)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()
                    
                    tracker.push_greater_of(1, node1)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()
                    
                else: # 1 and 2 are inner nodes
                    tracker.push_less_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1.less, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()
                    tracker.pop()
                        
                    tracker.push_greater_of(1, node1)
                    tracker.push_less_of(2, node2)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.less, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()
                        
                    tracker.push_greater_of(2, node2)
                    self.__wcount_neighbors_custom_2D_traverse(
                        other, n_queries, r, results, idx,
                        node1.greater, node2.greater, tracker,
                        period, sweights, oweights, w, saux, oaux, wdim)
                    tracker.pop()
                    tracker.pop()
                    
        return 0

    @cython.boundscheck(False)
    def wcount_neighbors_custom_2D(cKDTree self, cKDTree other, 
                                object r, np.float64_t p=2.,
                                object period = None, 
                                object sweights = None, object oweights = None,
                                Function w=None, object saux=None, object oaux=None,
                                wdim=1):
        """wcount_neighbors_custom(self, other, r, p)

        Weighted count of how many nearby pairs can be formed for each point in self.

        Count the number of pairs (x1,x2) can be formed, with x1 drawn
        from self and x2 drawn from `other`, and where
        ``distance(x1, x2, p) <= r``.
        This is the "two-point correlation" described in Gray and Moore 2000,
        "N-body problems in statistical learning", and the code here is based
        on their algorithm.

        Parameters
        ----------
        other : KDTree instance
            The other tree to draw points from.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use
        period : array_like, dimension self.m
            A vector indicating the periodic length along each dimension.
        sweights : array_like, dimension self.n
            A vector indicating the weight attached to each point in self.
        oweights : array_like, dimension other.n
            A vector indicating the weight attached to each point in other.
        w: ckdtree.Function object.  Function used in weighting.  None results in w1*w2
            w(self weight, other weight, self aux_data, other aux_data)
        saux: array_like, dimension self.n
            A vector indicating the an auxiliary weight attached to each point in self.
        oaux: array_like, dimension self.n
            A vector indicating the an auxiliary weight attached to each point in other.
        wdim: int
            The dimension of the result vector
        

        Returns
        -------
        result : float or 1-D array of floats
            The weighted number of pairs. Note that this is internally stored in a numpy float,
            and so may overflow if very large (2e9).

        """
        
        #process count function
        if w is None:
            w = jweight()
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        cdef np.intp_t n_queries, i
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] real_r
        cdef np.ndarray[np.intp_t, ndim=1, mode="c"] idx
        
        cdef np.intp_t cwdim = wdim
        
        #process the self weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] csweights #copy of self weights
        if sweights is None:
            sweights = np.array([1.0]*self.n, dtype=np.float64)
        else:
            sweights = np.asarray(sweights).astype("float64")
        csweights = np.ascontiguousarray(sweights)
        
        #process the other weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] coweights #copy of other weights
        if oweights is None:
            oweights = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oweights).astype("float64")
        coweights = np.ascontiguousarray(oweights)
        
        #process the self aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] csaux #copy of self weights
        if saux is None:
            saux = np.array([1.0]*self.n, dtype=np.float64)
        else:
            saux = np.asarray(saux).astype("float64")
        csaux = np.ascontiguousarray(saux)
        
        #process the other aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] coaux #copy of other weights
        if oaux is None:
            oaux = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oaux).astype("float64")
        coaux = np.ascontiguousarray(oaux)

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to count_neighbors have different dimensionality")

        # Make a copy of r array to ensure it's contiguous and to modify it below
        if np.shape(r) == ():
            real_r = np.array([r], dtype=np.float64)
            n_queries = 1
        elif len(np.shape(r))==1:
            real_r = np.array(r, dtype=np.float64)
            n_queries = r.shape[0]
        else:
            raise ValueError("r must be either a single value or a one-dimensional array of values")

        # Internally, we represent all distances as distance ** p
        if p != infinity:
            for i in range(n_queries):
                if real_r[i] != infinity:
                    real_r[i] = real_r[i] ** p

        # Track node-to-node min/max distances
        tracker = RectRectDistanceTracker(Rectangle(self.mins, self.maxes),
                                          Rectangle(other.mins, other.maxes),
                                          p, 0.0, 0.0, period)
        
        
        results = np.zeros((wdim*n_queries), dtype=np.float64).reshape((wdim,n_queries))
        cdef np.float64_t [:, :] results_view = results
        idx = np.arange(n_queries, dtype=np.intp)
        self.__wcount_neighbors_custom_2D_traverse(other, n_queries,
                                                &real_r[0], results_view, &idx[0],
                                                self.tree, other.tree,
                                                tracker,  <np.float64_t*>cperiod.data,
                                                <np.float64_t*>csweights.data,
                                                <np.float64_t*>coweights.data,
                                                w, 
                                                <np.float64_t*>csaux.data,
                                                <np.float64_t*>coaux.data, cwdim)

        if np.shape(r) == ():
            return results[:,0]
        elif len(np.shape(r))==1:
            return results
    
    
    # ----------------------
    # sparse_distance_matrix
    # ----------------------
    cdef int __sparse_distance_matrix_traverse(cKDTree self, cKDTree other, 
                                               coo_entries results,
                                               innernode* node1, innernode* node2,
                                               RectRectDistanceTracker tracker, 
                                               np.float64_t*period) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.float64_t d
        cdef np.intp_t i, j, min_j
                
        if tracker.min_distance > tracker.upper_bound:
            return 0
        elif node1.split_dim == -1:  # 1 is leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # 1 & 2 are leaves
                lnode2 = <leafnode*>node2
                
                
                # brute-force
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    # Special care here to avoid duplicate pairs
                    if node1 == node2:
                        min_j = i+1
                    else:
                        min_j = lnode2.start_idx
                
                    for j in range(min_j, lnode2.end_idx):
                        d = _distance_p_periodic(
                            self.raw_data + self.raw_indices[i] * self.m,
                            other.raw_data + other.raw_indices[j] * other.m,
                            tracker.p, self.m, tracker.upper_bound, period)
                        if d <= tracker.upper_bound:
                            if tracker.p != 1 and tracker.p != infinity:
                               d = d**(1.0 / tracker.p)
                            col = imax(self.raw_indices[i],other.raw_indices[j])
                            row = imin(self.raw_indices[i],other.raw_indices[j])
                            #results.add(self.raw_indices[i], other.raw_indices[j], d)
                            results.add(row, col, d)
                            #if node1 == node2:
                            #    #results.add(self.raw_indices[j], other.raw_indices[i], d)

            else:  # 1 is a leaf node, 2 is inner node
                tracker.push_less_of(2, node2)
                self.__sparse_distance_matrix_traverse(
                    other, results, node1, node2.less, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__sparse_distance_matrix_traverse(
                    other, results, node1, node2.greater, tracker, period)
                tracker.pop()
                
        else:  # 1 is an inner node
            if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                tracker.push_less_of(1, node1)
                self.__sparse_distance_matrix_traverse(
                    other, results, node1.less, node2, tracker, period)
                tracker.pop()
                
                tracker.push_greater_of(1, node1)
                self.__sparse_distance_matrix_traverse(
                    other, results, node1.greater, node2, tracker, period)
                tracker.pop()
                
            else: # 1 and 2 are inner nodes
                tracker.push_less_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__sparse_distance_matrix_traverse(
                    other, results, node1.less, node2.less, tracker, period)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__sparse_distance_matrix_traverse(
                    other, results, node1.less, node2.greater, tracker, period)
                tracker.pop()
                tracker.pop()
                    
                tracker.push_greater_of(1, node1)
                if node1 != node2:
                    # Avoid traversing (node1.less, node2.greater) and
                    # (node1.greater, node2.less) (it's the same node pair
                    # twice over, which is the source of the complication in
                    # the original KDTree.sparse_distance_matrix)
                    tracker.push_less_of(2, node2)
                    self.__sparse_distance_matrix_traverse(
                        other, results, node1.greater, node2.less, tracker, period)
                    tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__sparse_distance_matrix_traverse(
                    other, results, node1.greater, node2.greater, tracker, period)
                tracker.pop()
                tracker.pop()
                
        return 0
            
    def sparse_distance_matrix(cKDTree self, cKDTree other, np.float64_t max_distance,
                               np.float64_t p=2.0, object period = None):
        """
        sparse_distance_matrix(self, other, max_distance, p=2.0)

        Compute a sparse distance matrix

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : cKDTree

        max_distance : positive float
        
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use. 

        Returns
        -------
        result : dok_matrix
            Sparse matrix representing the results in "dictionary of keys" format.
            FIXME: Internally, built as a COO matrix, it would be more
            efficient to return this COO matrix.

        """

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to sparse_distance_matrix have different dimensionality")
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)

        # Calculate mins and maxes to outer box
        tracker = RectRectDistanceTracker(Rectangle(self.mins, self.maxes),
                                          Rectangle(other.mins, other.maxes),
                                          p, 0, max_distance, period)
        
        results = coo_entries()
        self.__sparse_distance_matrix_traverse(other, results, self.tree, other.tree,
                                               tracker, <np.float64_t*>cperiod.data)
        
        return results.to_matrix(shape=(self.n, other.n))


    # ----------------------
    # custom sparse_distance_matrix
    # ----------------------
    cdef int __sparse_distance_matrix_traverse_custom(cKDTree self, cKDTree other, 
                                               coo_entries results,
                                               innernode* node1, innernode* node2,
                                               RectRectDistanceTracker tracker, 
                                               np.float64_t*period, np.float64_t*los,
                                               np.float64_t b_para, np.float64_t b_perp,
                                               np.float64_t*sweights,
                                               np.float64_t*oweights,
                                               Function w,
                                               np.float64_t*saux,
                                               np.float64_t*oaux) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.float64_t d, d_para, d_perp
        cdef np.intp_t i, j, min_j
                
        if tracker.min_distance > tracker.upper_bound:
            return 0
        elif node1.split_dim == -1:  # 1 is leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # 1 & 2 are leaves
                lnode2 = <leafnode*>node2
                
                
                # brute-force
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    # Special care here to avoid duplicate pairs
                    if node1 == node2:
                        min_j = i+1
                    else:
                        min_j = lnode2.start_idx
                
                    for j in range(min_j, lnode2.end_idx):
                        d_para, d_perp = _projected_distance_p_periodic(
                            self.raw_data + self.raw_indices[i] * self.m,
                            other.raw_data + other.raw_indices[j] * other.m,
                            tracker.p, self.m, period, los)
                        d = d_para+d_perp
                        if d <= tracker.upper_bound:
                            d_para = d_para**(1.0 / tracker.p)
                            d_perp = d_perp**(1.0 / tracker.p)
                            link = (d_para/b_para)*(d_para/b_para)+(d_perp/b_perp)*(d_perp/b_perp)
                            if link<=1.0:
                                if sweights[self.raw_indices[i]]==sweights[other.raw_indices[j]]:
                                    d = d**(1.0 / tracker.p)
                                    #col = imax(self.raw_indices[i],other.raw_indices[j])
                                    #row = imin(self.raw_indices[i],other.raw_indices[j])
                                    #results.add(row, col, d)
                                    results.add(self.raw_indices[i], other.raw_indices[j], d)
                                    if node1 == node2:
                                        results.add(self.raw_indices[j], other.raw_indices[i], d)

            else:  # 1 is a leaf node, 2 is inner node
                tracker.push_less_of(2, node2)
                self.__sparse_distance_matrix_traverse_custom(
                    other, results, node1, node2.less, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__sparse_distance_matrix_traverse_custom(
                    other, results, node1, node2.greater, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                tracker.pop()
                
        else:  # 1 is an inner node
            if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                tracker.push_less_of(1, node1)
                self.__sparse_distance_matrix_traverse_custom(
                    other, results, node1.less, node2, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                tracker.pop()
                
                tracker.push_greater_of(1, node1)
                self.__sparse_distance_matrix_traverse_custom(
                    other, results, node1.greater, node2, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                tracker.pop()
                
            else: # 1 and 2 are inner nodes
                tracker.push_less_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__sparse_distance_matrix_traverse_custom(
                    other, results, node1.less, node2.less, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__sparse_distance_matrix_traverse_custom(
                    other, results, node1.less, node2.greater, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                tracker.pop()
                tracker.pop()
                    
                tracker.push_greater_of(1, node1)
                if node1 != node2:
                    # Avoid traversing (node1.less, node2.greater) and
                    # (node1.greater, node2.less) (it's the same node pair
                    # twice over, which is the source of the complication in
                    # the original KDTree.sparse_distance_matrix)
                    tracker.push_less_of(2, node2)
                    self.__sparse_distance_matrix_traverse_custom(
                        other, results, node1.greater, node2.less, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                    tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__sparse_distance_matrix_traverse_custom(
                    other, results, node1.greater, node2.greater, tracker, period, los, b_para, b_perp, sweights, oweights, w, saux, oaux)
                tracker.pop()
                tracker.pop()
                
        return 0
            
    def sparse_distance_matrix_custom(cKDTree self, cKDTree other, np.float64_t max_distance,
                               np.float64_t p=2.0, object period = None, object los=None,
                               np.float64_t b_para=1.0, np.float64_t b_perp=1.0,
                               object sweights = None, object oweights = None,
                               Function w=None, object saux=None, object oaux=None):
        """
        sparse_distance_matrix(self, other, max_distance, p=2.0)

        Compute a sparse distance matrix

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : cKDTree

        max_distance : positive float
        
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use. 

        Returns
        -------
        result : dok_matrix
            Sparse matrix representing the results in "dictionary of keys" format.
            FIXME: Internally, built as a COO matrix, it would be more
            efficient to return this COO matrix.

        """

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to sparse_distance_matrix have different dimensionality")
        
        #process the period parameter
        cdef np.ndarray[np.float64_t, ndim=1] cperiod
        if period is None:
            period = np.array([np.inf]*self.m)
        else:
            period = np.asarray(period).astype("float64")
        cperiod = np.ascontiguousarray(period)
        
        #process the los parameter
        cdef np.ndarray[np.float64_t, ndim=1] clos
        if los is None:
            los = np.zeros(self.m)
            los[self.m-1]=1.0
        else:
            los = np.asarray(los).astype("float64")
        clos = np.ascontiguousarray(los)
        
        #process count function
        if w is None:
            w = fmultiply()
        
        #process the self weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] csweights #copy of self weights
        if sweights is None:
            sweights = np.array([1.0]*self.n, dtype=np.float64)
        else:
            sweights = np.asarray(sweights).astype("float64")
        csweights = np.ascontiguousarray(sweights)
        
        #process the other weights parameter
        cdef np.ndarray[np.float64_t, ndim=1] coweights #copy of other weights
        if oweights is None:
            oweights = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oweights).astype("float64")
        coweights = np.ascontiguousarray(oweights)
        
        #process the self aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] csaux #copy of self weights
        if saux is None:
            saux = np.array([1.0]*self.n, dtype=np.float64)
        else:
            saux = np.asarray(saux).astype("float64")
        csaux = np.ascontiguousarray(saux)
        
        #process the other aux parameter
        cdef np.ndarray[np.float64_t, ndim=1] coaux #copy of other weights
        if oaux is None:
            oaux = np.array([1.0]*other.n, dtype=np.float64)
        else:
            oweights = np.asarray(oaux).astype("float64")
        coaux = np.ascontiguousarray(oaux)

        # Calculate mins and maxes to outer box
        tracker = RectRectDistanceTracker(Rectangle(self.mins, self.maxes),
                                          Rectangle(other.mins, other.maxes),
                                          p, 0, max_distance, period)
        
        results = coo_entries()
        self.__sparse_distance_matrix_traverse_custom(other, results, self.tree, other.tree,
                                               tracker, <np.float64_t*>cperiod.data,
                                               <np.float64_t*>clos.data, b_para, b_perp,
                                               <np.float64_t*>csweights.data,
                                               <np.float64_t*>coweights.data,
                                               w, 
                                               <np.float64_t*>csaux.data,
                                               <np.float64_t*>coaux.data)
        
        return results.to_matrix(shape=(self.n, other.n))


cdef class Function:
    cpdef double evaluate(self, double x, double y, double a, double b) except *:
        return 0

cdef class fmultiply(Function):
    cpdef double evaluate(self, double x, double y, double a, double b) except *:
        return x * y

cdef class jweight(Function):
    cpdef double evaluate(self, double x, double y, double a, double b) except *:
        if a==0: return 1.0
        elif (x==y) & (x==a): return 0.0 # both outside the sub-sample
        elif x==y: return 1.0 # both inside the sub-sample        
        elif (x!=y) & ((x==a) or (y==a)): return 0.5 # only one inside the sub-sample
        elif (x!=y) & (x!=a) & (y!=a): return 1.0 # both inside the sub-sample
        
