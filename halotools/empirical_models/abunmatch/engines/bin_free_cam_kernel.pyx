# cython: language_level=2
"""
"""
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor
import numpy as np
cimport cython
from ....utils import unsorting_indices

__all__ = ('cython_bin_free_cam_kernel', 'get_value_at_rank')


cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int _bisect_left_kernel(double[:] arr, double value):
    """ Return the index where to insert ``value`` in list ``arr`` of length ``n``,
    assuming ``arr`` is sorted.

    This function is equivalent to the bisect_left function implemented in the
    python standard libary bisect.
    """
    cdef int n = arr.shape[0]
    cdef int ifirst_subarr = 0
    cdef int ilast_subarr = n
    cdef int imid_subarr

    while ilast_subarr-ifirst_subarr >= 2:
        imid_subarr = (ifirst_subarr + ilast_subarr)/2
        if value > arr[imid_subarr]:
            ifirst_subarr = imid_subarr
        else:
            ilast_subarr = imid_subarr
    if value > arr[ifirst_subarr]:
        return ilast_subarr
    else:
        return ifirst_subarr


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void _insert_first_pop_last_kernel(int* arr, int value_in, int n):
    """ Insert the element ``value_in`` into the input array and pop out the last element
    """
    cdef int i
    for i in range(n-2, -1, -1):
        arr[i+1] = arr[i]
    arr[0] = value_in


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int _correspondence_indices_shift(int idx_in, int idx_out, int idx):
    """ Update the correspondence indices array
    """
    cdef int shift = 0
    if idx_in < idx_out:
        if idx_in <= idx < idx_out:
            shift = 1
    elif idx_in > idx_out:
        if idx_out < idx <= idx_in:
            shift = -1
    return shift


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void _insert_pop_kernel(double* arr, int idx_in, int idx_out, double value_in):
    """ Pop out the value stored in index ``idx_out`` of array ``arr``,
    and insert ``value_in`` at index ``idx_in`` of the final array.
    """
    cdef int i

    if idx_in <= idx_out:
        for i in range(idx_out-1, idx_in-1, -1):
            arr[i+1] = arr[i]
    else:
        for i in range(idx_out, idx_in):
            arr[i] = arr[i+1]
    arr[idx_in] = value_in


def setup_initial_indices(int iy, int nwin, int npts):
    """ Search an array of length npts to identify
    the unique window of width nwin centered iy. Care is taken to deal with
    the left and right edges. For elements iy < nwin/2, the first nwin elements
    of the array are used; for elements iy > npts-nwin/2,
    the last nwin elements are used.
    """
    cdef int nhalfwin = int(nwin/2)
    cdef int init_iy_low = iy - nhalfwin
    cdef int init_iy_high = init_iy_low+nwin

    if init_iy_low < 0:
        init_iy_low = 0
        init_iy_high = init_iy_low + nwin
        iy = init_iy_low + nhalfwin
    elif init_iy_high > npts - nhalfwin:
        init_iy_high = npts
        init_iy_low = init_iy_high - nwin
        iy = init_iy_low + nhalfwin

    return iy, init_iy_low, init_iy_high


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int _find_index(int[:] arr, int val):
    for i in range(len(arr)):
        if arr[i] == val:
            return i
    return -1


# Public version of _get_value_at_rank as we need to call this from python.
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def get_value_at_rank(double[:] sorted_values, int rank1, int nwin, int add_subgrid_noise):
    return _get_value_at_rank(sorted_values, rank1, nwin, add_subgrid_noise)

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double _get_value_at_rank(double[:] sorted_values, int rank1, int nwin, int add_subgrid_noise):
    if add_subgrid_noise == 0:
        return sorted_values[rank1]
    else:
        low_rank = max(rank1 - 1, 0)
        high_rank = min(rank1 + 1, nwin - 1)
        low_cdf = sorted_values[low_rank]
        high_cdf = sorted_values[high_rank]
        return low_cdf + (high_cdf-low_cdf)*random_uniform()


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def cython_bin_free_cam_kernel(double[:] y1, double[:] y2, int[:] i2_match, int nwin,
            int add_subgrid_noise, int return_indexes):
    """ Kernel underlying the bin-free implementation of conditional abundance matching.
    For the i^th element of y1, we define a window of length `nwin` surrounding the
    point y1[i], and another window surrounding y2[i2_match[i]]. We calculate the
    rank-order of y1[i] within the first window. Then we find the point in the second
    window with a matching rank-order and use this value as ynew[i].
    The algorithm has been implemented so that the windows are only sorted once
    at the beginning, and as the windows slide along the arrays with increasing i,
    elements are popped in and popped out so as to preserve the sorted order.

    When using add_subgrid_noise, the algorithm differs slightly. Rather than setting
    ynew[i] to the value in the second window with the matching rank-order,
    instead we assign a random uniform number from the range spanned by
    (y2_window[rank-1],y2_window[rank+1]). This reduces discreteness effects
    and comes at no loss of precision since the PDF is not known to an accuracy
    better than 1/nwin.

    The arrays named sorted_cdf_values store the y-values in the two windows.
    The arrays correspondence_indx are responsible for the bookkeeping involved in
    maintaining a sorted order as elements are popped in and popped out.
    The way this works is that as the window slides along from left to right,
    the y value corresponding to the smallest x in the window should be
    the next one popped out. However, the position of this element within
    sorted_cdf_values can be anywhere, since there is no strict connection
    between the values of x and y. So the correspondence_indx array is used
    to keep track of the x-ordering of the y-values stored in the sorted_cdf_values
    windows. In particular, the value of the first element in correspondence_indx
    stores the position of the most-recently added element to sorted_cdf_values
    (i.e., the element with the largest x); the last element in correspondence_indx,
    element nwin-1, stores the position of the element of sorted_cdf_values
    that will be the next one popped out (i.e., the element with the largest x);
    the middle element of correspondence_indx, element nwin/2,
    stores the position of the middle of the sorted_cdf_values window.
    Since the position within sorted_cdf_values is the rank,
    then sorted_cdf_values[correspondence_indx[nwin/2]] stores the value of ynew[i].

    """
    cdef int nhalfwin = int(nwin/2)
    cdef int npts1 = y1.shape[0]
    cdef int npts2 = y2.shape[0]

    cdef int iy1, i, j, idx, idx2, iy2_match
    cdef int idx_in1, idx_out1, idx_in2, idx_out2
    cdef double value_in1, value_out1, value_in2, value_out2

    cdef int[:] y1_new_indexes = np.zeros(npts1, dtype='i4') - 1
    cdef double[:] y1_new_values = np.zeros(npts1, dtype='f8') - 1

    cdef int rank1, rank2

    #  Set up initial window arrays for y1
    cdf_values1 = np.copy(y1[:nwin])
    idx_sorted_cdf_values1 = np.argsort(cdf_values1)
    cdef double[:] sorted_cdf_values1 = np.ascontiguousarray(
        cdf_values1[idx_sorted_cdf_values1], dtype='f8')
    cdef int[:] correspondence_indx1 = np.ascontiguousarray(
        unsorting_indices(idx_sorted_cdf_values1)[::-1], dtype='i4')

    #  Set up initial window arrays for y2
    cdef int iy2_init = i2_match[nhalfwin]
    _iy2, init_iy2_low, init_iy2_high = setup_initial_indices(
                iy2_init, nwin, npts2)
    cdef int iy2 = _iy2
    cdef int iy2_max = npts2 - nhalfwin - 1

    cdef int low_rank, high_rank
    cdef int index
    cdef double low_cdf, high_cdf

    #  Ensure that any bookkeeping error in setting up the window
    #  is caught by an exception rather than a bus error
    msg = ("Bookkeeping error internal to cython_bin_free_cam_kernel\n"
        "init_iy2_low = {0}, init_iy2_high = {1}, nwin = {2}")
    assert init_iy2_high - init_iy2_low == nwin, msg.format(
            init_iy2_low, init_iy2_high, nwin)

    cdf_values2 = np.copy(y2[init_iy2_low:init_iy2_high])

    idx_sorted_cdf_values2 = np.argsort(cdf_values2)

    cdef double[:] sorted_cdf_values2 = np.ascontiguousarray(
        cdf_values2[idx_sorted_cdf_values2], dtype='f8')
    cdef int[:] correspondence_indx2 = np.ascontiguousarray(
        unsorting_indices(idx_sorted_cdf_values2)[::-1], dtype='i4')

    #  Loop over elements of the first array, ignoring the first and last nwin/2 points,
    #  which will be treated separately by the python wrapper.
    for iy1 in range(nhalfwin, npts1-nhalfwin):

        rank1 = correspondence_indx1[nhalfwin]
        #  Where to center the second window (making sure we don't slide off the end)
        iy2_match = min(i2_match[iy1], iy2_max)

        #  Continue to slide the window along the second array
        #  until we find the matching point, updating the window with each iteration
        while iy2 < iy2_match:

            #  Find the value coming in and the value coming out
            value_in2 = y2[iy2 + nhalfwin + 1]
            idx_out2 = correspondence_indx2[nwin-1]
            value_out2 = sorted_cdf_values2[idx_out2]

            #  Find the position where we will insert the new point into the second window
            idx_in2 = _bisect_left_kernel(sorted_cdf_values2, value_in2)
            if value_in2 > value_out2:
                idx_in2 -= 1

            #  Update the correspondence array
            _insert_first_pop_last_kernel(&correspondence_indx2[0], idx_in2, nwin)
            for j in range(1, nwin):
                idx2 = correspondence_indx2[j]
                correspondence_indx2[j] += _correspondence_indices_shift(
                    idx_in2, idx_out2, idx2)

            #  Update the CDF window
            _insert_pop_kernel(&sorted_cdf_values2[0], idx_in2, idx_out2, value_in2)

            iy2 += 1

        #  The array sorted_cdf_values2 is now centered on the correct point of y2
        #  We have already calculated the rank-order of the point iy1, rank1
        #  So we either directly map sorted_cdf_values2[rank1] to ynew,
        #  or alternatively we randomly draw a value between
        #  sorted_cdf_values2[rank1-1] and sorted_cdf_values2[rank1+1]
        if return_indexes == 1:
            index = _find_index(correspondence_indx2, rank1)
            if index == -1:
                raise Exception("Index {} not found in correspondence_indx2".format(rank1))
            y1_new_indexes[iy1] = iy2 + nhalfwin - index
        else:
            y1_new_values[iy1] = _get_value_at_rank(sorted_cdf_values2, rank1, nwin, add_subgrid_noise)

        #  Move on to the next value in y1

        #  Find the value coming in and the value coming out
        value_in1 = y1[iy1 + nhalfwin + 1]
        idx_out1 = correspondence_indx1[nwin-1]
        value_out1 = sorted_cdf_values1[idx_out1]

        #  Find the position where we will insert the new point into the first window
        idx_in1 = _bisect_left_kernel(sorted_cdf_values1, value_in1)
        if value_in1 > value_out1:
            idx_in1 -= 1

        #  Update the correspondence array
        _insert_first_pop_last_kernel(&correspondence_indx1[0], idx_in1, nwin)
        for i in range(1, nwin):
            idx = correspondence_indx1[i]
            correspondence_indx1[i] += _correspondence_indices_shift(
                idx_in1, idx_out1, idx)

        #  Update the CDF window
        _insert_pop_kernel(&sorted_cdf_values1[0], idx_in1, idx_out1, value_in1)

    if return_indexes:
        return y1_new_indexes
    return y1_new_values
