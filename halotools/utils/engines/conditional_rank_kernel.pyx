# cython: language_level=2
"""
"""
import numpy as np
cimport cython

from ..array_utils import unsorting_indices

__all__ = ('cython_conditional_rank_kernel', )


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


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def cython_conditional_rank_kernel(double[:] y_sorted, int nwin):
    """
    """
    cdef int nhalfwin = int(nwin/2)
    cdef int iy, idx_in, idx_out, idx_temp, i, idx
    cdef double value_in, value_out
    cdef int npts = y_sorted.shape[0]
    cdef double[:] result = np.zeros(npts, dtype='f8')

    cdef int ifirst_subarr, ilast_subarr, imid_subarr, shift

    cdf_value_table = np.copy(y_sorted[:nwin])
    idx_sorted_cdf_values = np.argsort(cdf_value_table)
    _sorted_cdf_value_table = np.copy(cdf_value_table[idx_sorted_cdf_values])
    _correspondence_indices = np.copy(unsorting_indices(idx_sorted_cdf_values)[::-1])

    cdef double[:] sorted_cdf_value_table = np.array(_sorted_cdf_value_table, dtype='f8')
    cdef int[:] correspondence_indices = np.array(_correspondence_indices, dtype='i4')

    for iy in range(nhalfwin, npts-nhalfwin-1):
        result[iy] = correspondence_indices[nhalfwin]
        value_in = y_sorted[iy + nhalfwin + 1]

        idx_out = correspondence_indices[nwin-1]
        value_out = sorted_cdf_value_table[idx_out]

        if value_in <= value_out:
            idx_in = _bisect_left_kernel(sorted_cdf_value_table, value_in)
        else:
            idx_in = _bisect_left_kernel(sorted_cdf_value_table, value_in) - 1

        _insert_first_pop_last_kernel(&correspondence_indices[0], idx_in, nwin)

        for i in range(1, nwin):
            idx = correspondence_indices[i]
            correspondence_indices[i] += _correspondence_indices_shift(idx_in, idx_out, idx)

        _insert_pop_kernel(&sorted_cdf_value_table[0], idx_in, idx_out, value_in)

    return result
