r"""
Module containing the `~halotools.mock_observables.marked_tpcf` function used to
calculate the marked two-point correlation function.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .clustering_helpers import process_optional_input_sample2

from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_separation_bins_array, get_period, get_num_threads)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length

from ..pair_counters import npairs_3d, marked_npairs_3d

from ...custom_exceptions import HalotoolsError


__all__ = ['marked_tpcf']
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def marked_tpcf(sample1, rbins, sample2=None,
        marks1=None, marks2=None, period=None, do_auto=True, do_cross=True,
        num_threads=1, weight_func_id=1,
        normalize_by='random_marks', iterations=1, randomize_marks=None, seed=None):
    r"""
    Calculate the real space marked two-point correlation function, :math:`\mathcal{M}(r)`.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function. Default is None, in which case only the
        auto-correlation function will be calculated.

    marks1 : array_like, optional
        len(sample1) x N_marks array of marks.  The supplied marks array must have the
        appropriate shape for the chosen ``weight_func_id`` (see Notes for requirements).  If this
        parameter is not specified, it is set to numpy.ones((len(sample1), N_marks)).

    marks2 : array_like, optional
        len(sample2) x N_marks array of marks.  The supplied marks array must have the
        appropriate shape for the chosen ``weight_func_id`` (see Notes for requirements).  If this
        parameter is not specified, it is set to numpy.ones((len(sample2), N_marks)).

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    do_auto : boolean, optional
        Boolean determines whether the auto-correlation function will
        be calculated and returned. Default is True.

    do_cross : boolean, optional
        Boolean determines whether the cross-correlation function will
        be calculated and returned. Only relevant when ``sample2`` is also provided.
        Default is True for the case where ``sample2`` is provided, otherwise False.

    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed
        using the python ``multiprocessing`` module. Default is 1 for a purely serial
        calculation, in which case a multiprocessing Pool object will
        never be instantiated. A string 'max' may be used to indicate that
        the pair counters should use all available cores on the machine.

    weight_func_id : int, optional
        Integer ID indicating which marking function should be used.
        See notes for a list of available marking functions.

    normalize_by : string, optional
        A string indicating how to normailze the weighted pair counts in the marked
        correlation function calculation.  Options are: 'random_marks' or 'number_counts'.
        See Notes for more detail.

    iterations : int, optional
        integer indicating the number of times to calculate the random weights,
        taking the mean of the outcomes.  Only applicable if ``normalize_by`` is set
        to 'random_marks'.  See Notes for further explanation.

    randomize_marks : array_like, optional
        Boolean array of length N_marks indicating which elements should be randomized
        when calculating the random weighted pair counts.  Default is [True]*N_marks.
        This parameter is only applicable if ``normalize_by`` is set to 'random_marks'.
        See Notes for more detail.

    seed : int, optional
        Random number seed used to shuffle the marks
        and to randomly downsample data, if applicable.
        Default is None, in which case downsampling and shuffling will be stochastic.

    Returns
    -------
    marked_correlation_function(s) : numpy.array
        *len(rbins)-1* length array containing the marked correlation function
        :math:`\mathcal{M}(r)` computed in each of the bins defined by ``rbins``.

        .. math::
            \mathcal{M}(r) \equiv \mathrm{WW}(r) / \mathrm{XX}(r),

        where :math:`\mathrm{WW}(r)` is the weighted number of pairs with separations
        equal to :math:`r`, and :math:`\mathrm{XX}(r)` is dependent on the choice of the
        ``normalize_by`` parameter.  If ``normalize_by`` is 'random_marks'
        :math:`XX \equiv \mathcal{RR}`, the weighted pair counts where the marks have
        been randomized marks.  If ``normalize_by`` is 'number_counts'
        :math:`XX \equiv DD`, the unweighted pair counts.
        See Notes for more detail.

        If ``sample2`` is passed as input, three arrays of length *len(rbins)-1* are
        returned:

        .. math::
            \mathcal{M}_{11}(r), \ \mathcal{M}_{12}(r), \ \mathcal{M}_{22}(r),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and
        ``sample2``, and the autocorrelation of ``sample2``.  If ``do_auto`` or
        ``do_cross`` is set to False, the appropriate result(s) is not returned.

    Notes
    -----
    Pairs are counted using
    `~halotools.mock_observables.pair_counters.marked_npairs_3d`.

    If the ``period`` argument is passed in, the ith coordinate of all points
    must be between 0 and period[i].

    ``normalize_by`` indicates how to calculate :math:`\mathrm{XX}`.  If ``normalize_by``
    is 'random_marks', then :math:`\mathrm{XX} \equiv \mathcal{RR}`, and
    :math:`\mathcal{RR}` is calculated by randomizing the marks among points according
    to the ``randomize_marks`` mask.  This marked correlation function is then:

    .. math::
        \mathcal{M}(r) \equiv \frac{\sum_{ij}f(m_i,m_j)}{\sum_{kl}f(m_k,m_l)}

    where the sum in the numerator is of pairs :math:`i,j` with separation :math:`r`,
    and marks :math:`m_i,m_j`.  :math:`f()` is the marking function, ``weight_func_id``.  The sum
    in the denominator is over an equal number of random pairs :math:`k,l`. The
    calculation of this sum can be done multiple times, by setting the ``iterations``
    parameter. The mean of the sum is then taken amongst iterations and used in the
    calculation.

    If ``normalize_by`` is 'number_counts', then :math:`\mathrm{XX} \equiv \mathrm{DD}`
    is calculated by counting total number of pairs using
    `~halotools.mock_observables.pair_counters.npairs_3d`.
    This is:

    .. math::
        \mathcal{M}(r) \equiv \frac{\sum_{ij}f(m_i,m_j)}{\sum_{ij} 1},

    There are multiple marking functions available.  In general, each requires a different
    number of marks per point, N_marks.  The marking function gets passed two vectors
    per pair, w1 and w2, of length N_marks and return a float.  The available marking
    functions, ``weight_func_id`` and the associated integer ID numbers are:

    #. multiplicaitive weights (N_marks = 1)
        .. math::
            f(w_1,w_2) = w_1[0] \times w_2[0]

    #. summed weights (N_marks = 1)
        .. math::
            f(w_1,w_2) = w_1[0] + w_2[0]

    #. equality weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_1[1]\times w_2[1] & : w_1[0] = w_2[0] \\
                    0.0 & : w_1[0] \neq w_2[0] \\
                \end{array}
                \right.

    #. inequality weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_1[1]\times w_2[1] & : w_1[0] \neq w_2[0] \\
                    0.0 & : w_1[0] = w_2[0] \\
                \end{array}
                \right.

    #. greater than weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_1[1]\times w_2[1] & : w_2[0] > w_1[0] \\
                    0.0 & : w_2[0] \leq w_1[0] \\
                \end{array}
                \right.

    #. less than weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_1[1]\times w_2[1] & : w_2[0] < w_1[0] \\
                    0.0 & : w_2[0] \geq w_1[0] \\
                \end{array}
                \right.

    #. greater than tolerance weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_2[1] & : w_2[0]>(w_1[0]+w_1[1]) \\
                    0.0 & : w_2[0] \leq (w_1[0]+w_1[1]) \\
                \end{array}
                \right.

    #. less than tolerance weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_2[1] & : w_2[0]<(w_1[0]+w_1[1]) \\
                    0.0 & : w_2[0] \geq (w_1[0]+w_1[1]) \\
                \end{array}
                \right.

    #. tolerance weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_2[1] & : |w_1[0]-w_2[0]|<w_1[1] \\
                    0.0 & : |w_1[0]-w_2[0]| \geq w_1[1] \\
                \end{array}
                \right.

    #. exclusion weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_2[1] & : |w_1[0]-w_2[0]|>w_1[1] \\
                    0.0 & : |w_1[0]-w_2[0]| \leq w_1[1] \\
                \end{array}
                \right.

    #. ratio weights (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    w_2[1] & : w2[0] > w1[0]*w1[1] \\
                    0.0 & : otherwise \\
                \end{array}
                \right.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    We transform our *x, y, z* points into the array shape used by the function by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    Assign random floats in the range [0,1] to the points to use as the marks:

    >>> marks = np.random.random(Npts)

    Use the multiplicative marking function:

    >>> rbins = np.logspace(-2,-1,10)
    >>> MCF = marked_tpcf(coords, rbins, marks1=marks, period=period, normalize_by='number_counts', weight_func_id=1)

    The result should be consistent with :math:`\langle {\rm mark}\rangle^2` at all *r*
    within the statistical errors.
    """

    # process parameters
    function_args = (sample1, rbins, sample2, marks1, marks2,
        period, do_auto, do_cross, num_threads,
        weight_func_id, normalize_by, iterations, randomize_marks, seed)
    sample1, rbins, sample2, marks1, marks2, period, do_auto, do_cross, num_threads,\
        weight_func_id, normalize_by, _sample1_is_sample2, PBCs,\
        randomize_marks = _marked_tpcf_process_args(*function_args)

    # calculate marked pairs
    W1W1, W1W2, W2W2 = marked_pair_counts(sample1, sample2, rbins, period,
        num_threads, do_auto, do_cross, marks1, marks2, weight_func_id, _sample1_is_sample2)

    if normalize_by == 'number_counts':
        R1R1, R1R2, R2R2 = pair_counts(sample1, sample2, rbins, period,
            num_threads, do_auto, do_cross, _sample1_is_sample2, None, None)
    # calculate randomized marked pairs
    elif normalize_by == 'random_marks':
        if iterations > 1:
            # create storage arrays of the right shape
            R1R1 = np.zeros((iterations, len(rbins)-1))
            R1R2 = np.zeros((iterations, len(rbins)-1))
            R2R2 = np.zeros((iterations, len(rbins)-1))
            for i in range(iterations):
                # get arrays to randomize marks
                with NumpyRNGContext(seed):
                    permutate1 = np.random.permutation(np.arange(0, len(sample1)))
                    permutate2 = np.random.permutation(np.arange(0, len(sample2)))
                R1R1[i, :], R1R2[i, :], R2R2[i, :] = random_counts(
                    sample1, sample2, rbins, period, num_threads,
                    do_auto, do_cross, marks1, marks2, weight_func_id,
                    _sample1_is_sample2, permutate1, permutate2, randomize_marks)

            R1R1 = np.median(R1R1, axis=0)
            R1R2 = np.median(R1R2, axis=0)
            R2R2 = np.median(R2R2, axis=0)
        else:
            # get arrays to randomize marks
            with NumpyRNGContext(seed):
                permutate1 = np.random.permutation(np.arange(0, len(sample1)))
                permutate2 = np.random.permutation(np.arange(0, len(sample2)))
            R1R1, R1R2, R2R2 = random_counts(sample1, sample2, rbins, period,
                num_threads, do_auto, do_cross, marks1, marks2, weight_func_id,
                _sample1_is_sample2, permutate1, permutate2, randomize_marks)

    # return results
    if _sample1_is_sample2:
        M_11 = W1W1/R1R1
        return M_11
    else:
        if (do_auto is True) & (do_cross is True):
            M_11 = W1W1/R1R1
            M_12 = W1W2/R1R2
            M_22 = W2W2/R2R2
            return M_11, M_12, M_22
        elif (do_cross is True):
            M_12 = W1W2/R1R2
            return M_12
        elif (do_auto is True):
            M_11 = W1W1/R1R1
            M_22 = W2W2/R2R2
            return M_11, M_22


def marked_pair_counts(sample1, sample2, rbins, period, num_threads,
        do_auto, do_cross, marks1, marks2, weight_func_id, _sample1_is_sample2):
    """
    Count weighted data pairs.
    """

    if do_auto is True:
        D1D1 = marked_npairs_3d(sample1, sample1, rbins,
            weights1=marks1, weights2=marks1,
            weight_func_id=weight_func_id, period=period, num_threads=num_threads)
        D1D1 = np.diff(D1D1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = marked_npairs_3d(sample1, sample2, rbins,
                weights1=marks1, weights2=marks2, weight_func_id=weight_func_id,
                period=period, num_threads=num_threads)
            D1D2 = np.diff(D1D2)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = marked_npairs_3d(sample2, sample2, rbins,
                weights1=marks2, weights2=marks2, weight_func_id=weight_func_id,
                period=period, num_threads=num_threads)
            D2D2 = np.diff(D2D2)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2


def random_counts(sample1, sample2, rbins, period, num_threads,
        do_auto, do_cross, marks1, marks2, weight_func_id,
        _sample1_is_sample2, permutate1, permutate2, randomize_marks):
    """
    Count random weighted data pairs.
    """

    permuted_marks1 = marks1
    permuted_marks2 = marks2
    for i in range(marks1.shape[1]):
        if randomize_marks[i]:
            permuted_marks1[:, i] = marks1[permutate1, i]
    for i in range(marks2.shape[1]):
        if randomize_marks[i]:
            permuted_marks2[:, i] = marks2[permutate2, i]

    if do_auto is True:
        R1R1 = marked_npairs_3d(sample1, sample1, rbins,
            weights1=marks1, weights2=permuted_marks1,
            weight_func_id=weight_func_id, period=period, num_threads=num_threads)
        R1R1 = np.diff(R1R1)
    else:
        R1R1 = None
        R2R2 = None

    if _sample1_is_sample2:
        R1R2 = R1R1
        R2R2 = R1R1
    else:
        if do_cross is True:
            R1R2 = marked_npairs_3d(sample1, sample2, rbins,
                weights1=permuted_marks1,
                weights2=permuted_marks2,
                weight_func_id=weight_func_id, period=period, num_threads=num_threads)
            R1R2 = np.diff(R1R2)
        else:
            R1R2 = None
        if do_auto is True:
            R2R2 = marked_npairs_3d(sample2, sample2, rbins,
                weights1=marks2, weights2=permuted_marks2,
                weight_func_id=weight_func_id, period=period, num_threads=num_threads)
            R2R2 = np.diff(R2R2)
        else:
            R2R2 = None

    return R1R1, R1R2, R2R2


def pair_counts(sample1, sample2, rbins, period, num_threads, do_auto, do_cross,
        _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
    """
    Count data-data pairs.
    """
    if do_auto is True:
        D1D1 = npairs_3d(sample1, sample1, rbins, period=period, num_threads=num_threads,
                      approx_cell1_size=approx_cell1_size,
                      approx_cell2_size=approx_cell1_size)
        D1D1 = np.diff(D1D1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_3d(sample1, sample2, rbins, period=period,
                          num_threads=num_threads,
                          approx_cell1_size=approx_cell1_size,
                          approx_cell2_size=approx_cell2_size)
            D1D2 = np.diff(D1D2)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = npairs_3d(sample2, sample2, rbins, period=period,
                          num_threads=num_threads,
                          approx_cell1_size=approx_cell2_size,
                          approx_cell2_size=approx_cell2_size)
            D2D2 = np.diff(D2D2)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2


def _marked_tpcf_process_args(sample1, rbins, sample2, marks1, marks2,
        period, do_auto, do_cross, num_threads,
        wfunc, normalize_by, iterations, randomize_marks, seed):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.marked_tpcf`.
    """

    sample1 = enforce_sample_has_correct_shape(sample1)

    sample2, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1, sample2, do_cross)

    # process wfunc parameter
    try:
        int(wfunc) == wfunc
    except:
        msg = ("\n `wfunc` parameter must be an integer ID of the desired function.")
        raise ValueError(msg)

    # process normalize_by parameter
    if normalize_by not in ['random_marks', 'number_counts']:
        msg = ("\n `normalize_by` parameter not recognized.")
        raise ValueError(msg)

    # process marks
    if marks1 is not None:
        marks1 = np.atleast_1d(marks1).astype(float)
    else:
        marks1 = np.ones(len(sample1)).astype(float)
    if marks2 is not None:
        marks2 = np.atleast_1d(marks2).astype(float)
    else:
        marks2 = np.ones(len(sample2)).astype(float)

    if marks1.ndim == 1:
        npts1 = len(marks1)
        marks1 = marks1.reshape((npts1, 1))
    elif marks1.ndim == 2:
        pass
    else:
        ndim1 = marks1.ndim
        msg = ("\n You must either pass in a 1-D or 2-D array \n"
               "for the input `marks1`. \n"
               "The `pair_counters._wnpairs_process_weights` function received \n"
               "a `marks1` array of dimension %i")
        raise HalotoolsError(msg % ndim1)

    if marks2.ndim == 1:
        npts2 = len(marks2)
        marks2 = marks2.reshape((npts2, 1))
    elif marks2.ndim == 2:
        pass
    else:
        ndim2 = marks2.ndim
        msg = ("\n You must either pass in a 1-D or 2-D array \n"
               "for the input `marks2`. \n"
               "The `pair_counters._wnpairs_process_weights` function received \n"
               "a `marks2` array of dimension %i")
        raise HalotoolsError(msg % ndim2)

    # check for consistency between marks and samples
    if len(marks1) != len(sample1):
        msg = ("\n `marks1` must have same length as `sample1`.")
        raise HalotoolsError(msg)
    if len(marks2) != len(sample2):
        msg = ("\n `marks2` must have same length as `sample2`.")
        raise HalotoolsError(msg)

    if randomize_marks is not None:
        randomize_marks = np.atleast_1d(randomize_marks)
    else:
        randomize_marks = np.array([True]*marks1.shape[1])

    if randomize_marks.ndim == 1:
        if len(randomize_marks) != marks1.shape[1]:
            msg = ("\n `randomize_marks` must have same length \n"
                   " as the number of weights per point.")
            raise HalotoolsError(msg)
    else:
        msg = ("\n `randomize_marks` must be one dimensional.")
        raise HalotoolsError(msg)

    rbins = get_separation_bins_array(rbins)
    rmax = np.amax(rbins)

    period, PBCs = get_period(period)

    _enforce_maximum_search_length(rmax, period)

    try:
        assert do_auto == bool(do_auto)
        assert do_cross == bool(do_cross)
    except:
        msg = "`do_auto` and `do_cross` keywords must be boolean-valued."
        raise ValueError(msg)

    num_threads = get_num_threads(num_threads)

    return sample1, rbins, sample2, marks1, marks2, period, do_auto, do_cross,\
        num_threads, wfunc, normalize_by, _sample1_is_sample2, PBCs, randomize_marks
