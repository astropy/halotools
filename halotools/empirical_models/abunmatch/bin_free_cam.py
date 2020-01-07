"""
"""
import numpy as np
from ...utils import unsorting_indices
from ...utils.conditional_percentile import _check_xyn_bounds, rank_order_function
from .engines import cython_bin_free_cam_kernel, get_value_at_rank
from .tests.naive_python_cam import sample2_window_indices


def conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=True,
            assume_x_is_sorted=False, assume_x2_is_sorted=False, return_indexes=False):
    r"""
    Given a set of input points with primary property `x` and secondary property `y`,
    and a mapping between that primary property and another secondary property
    (`y2 | x2`), assign values of the `y2` property to the input points.

    The `y2` that is assigned (`ynew`) is in monotonic correspondence with `y` at
    fixed `x`. Therefore, :math:`P(<y_{\rm new} | x) = P(<y | x)`.

    See :ref:`cam_tutorial` demonstrating how to use this function in galaxy-halo
    modeling with several worked examples.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (n1, ) storing the primary property of the input points.

    y : ndarray
        Numpy array of shape (n1, ) storing the secondary property of the input points.

    x2 : ndarray
        Numpy array of shape (n2, ) storing the primary property of the desired distribution. This should be the same physical property (e.g. halo mass) as x.

    y2 : ndarray
        Numpy array of shape (n2, ) storing the secondary property of the desired distribution. This is a different physical property to y.

    nwin : int
        Odd integer specifying the size of the window
        used to estimate :math:`P(<y | x)`. See Notes.

    add_subgrid_noise : bool, optional
        Flag determines whether random uniform noise will be added to fill in
        the gaps at the sub-grid level determined by `nwin`. This argument
        can be important for eliminating artificial discreteness effects.
        Default is True.

    assume_x_is_sorted : bool, optional
        Performance enhancement flag that can be used for cases where input `x` and `y`
        have been pre-sorted so that `x` is monotonically increasing. Default is False.

    assume_x2_is_sorted : bool, optional
        Performance enhancement flag that can be used for cases where input `x2` and `y2`
        have been pre-sorted so that `x2` is monotonically increasing. Default is False.

    return_indexes : bool, optional
        Return the indexes in y2 of where to find the new values, rather than the values.
        `add_subgrid_noise` must be set to False is this is set.
        Default is False.

    Returns
    -------
    ynew : ndarray
        Numpy array of shape (n1, ) storing the new values (or indexes if return_indexes is True) of
        the secondary property for the input points.

    Examples
    --------
    >>> npts1, npts2 = 5000, 3000
    >>> x = np.linspace(0, 1, npts1)
    >>> y = np.random.uniform(-1, 1, npts1)
    >>> x2 = np.linspace(0.5, 0.6, npts2)
    >>> y2 = np.random.uniform(-5, 3, npts2)
    >>> nwin = 51
    >>> new_y = conditional_abunmatch(x, y, x2, y2, nwin)

    Notes
    -----
    The ``nwin`` argument controls the precision of the calculation,
    and also the performance. For estimations of Prob(< y | x) with sub-percent accuracy,
    values of ``window_length`` must exceed 100. Values more tha a few hundred are
    likely overkill when using the (recommended) sub-grid noise option.

    With the release of Halotools v0.7, this function replaced a previous function
    of the same name. The old function is now called
    `~halotools.empirical_models.conditional_abunmatch_bin_based`.

    """
    if return_indexes:
        add_subgrid_noise = False

    x, y, nwin = _check_xyn_bounds(x, y, nwin)
    x2, y2, nwin = _check_xyn_bounds(x2, y2, nwin)
    nhalfwin = int(nwin/2)
    npts1 = len(x)

    if assume_x_is_sorted:
        x_sorted = x
        y_sorted = y
    else:
        idx_x_sorted = np.argsort(x)
        x_sorted = x[idx_x_sorted]
        y_sorted = y[idx_x_sorted]

    if assume_x2_is_sorted:
        x2_sorted = x2
        y2_sorted = y2
    else:
        idx_x2_sorted = np.argsort(x2)
        x2_sorted = x2[idx_x2_sorted]
        y2_sorted = y2[idx_x2_sorted]

    i2_matched = np.searchsorted(x2_sorted, x_sorted).astype('i4')

    result = np.array(cython_bin_free_cam_kernel(
        y_sorted, y2_sorted, i2_matched, nwin, int(add_subgrid_noise), int(return_indexes)))

    #  Finish the leftmost points in pure python
    iw = 0
    leftmost_window_ranks = rank_order_function(y_sorted[:nwin])
    for ix1 in range(0, nhalfwin):
        iy2_low, iy2_high = sample2_window_indices(ix1, x_sorted, x2_sorted, nwin)

        if return_indexes:
            leftmost_window_sorting_indexes = np.argsort(y2_sorted[iy2_low:iy2_high])
            result[ix1] = iy2_low + leftmost_window_sorting_indexes[
                    leftmost_window_ranks[iw]
            ]
        else:
            leftmost_sorted_window_y2 = np.sort(y2_sorted[iy2_low:iy2_high])
            result[ix1] = get_value_at_rank(leftmost_sorted_window_y2, leftmost_window_ranks[iw], nwin, int(add_subgrid_noise))

        iw += 1

    #  Finish the rightmost points in pure python
    iw = nhalfwin + 1
    rightmost_window_ranks = rank_order_function(y_sorted[-nwin:])
    for ix1 in range(npts1-nhalfwin, npts1):
        iy2_low, iy2_high = sample2_window_indices(ix1, x_sorted, x2_sorted, nwin)

        if return_indexes:
            rightmost_window_sorting_indexes = np.argsort(y2_sorted[iy2_low:iy2_high])
            result[ix1] = iy2_low + rightmost_window_sorting_indexes[
                    rightmost_window_ranks[iw]
            ]
        else:
            rightmost_sorted_window_y2 = np.sort(y2_sorted[iy2_low:iy2_high])
            result[ix1] = get_value_at_rank(rightmost_sorted_window_y2, rightmost_window_ranks[iw], nwin, int(add_subgrid_noise))
        iw += 1


    if return_indexes:
        # The result indexes point to the location in y2_sorted. Undo that if required
        result = result if assume_x2_is_sorted else idx_x2_sorted[result]
        # The result indexes are ordered like y_sorted. Undo that if required
        result = result if assume_x_is_sorted else result[unsorting_indices(idx_x_sorted)]
        return result
    else:
        # The result values are ordered like y_sorted, Undo that if required
        return result if assume_x_is_sorted else result[unsorting_indices(idx_x_sorted)]
