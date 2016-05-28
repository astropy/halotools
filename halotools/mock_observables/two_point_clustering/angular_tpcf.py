"""
Module containing the `~halotools.mock_observables.angular_tpcf` function used to
calculate galaxy clustering as a function of angular separation.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from warnings import warn

from .tpcf_estimators import _TP_estimator_requirements, _TP_estimator
from .clustering_helpers import verify_tpcf_estimator

from ..pair_counters import npairs_3d
from ..mock_observables_helpers import get_num_threads

from ...utils.spherical_geometry import spherical_to_cartesian, chord_to_cartesian
from ...custom_exceptions import HalotoolsError
from ...utils.array_utils import array_is_monotonic

__all__=['angular_tpcf']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def angular_tpcf(sample1, theta_bins, sample2=None, randoms=None,
        do_auto=True, do_cross=True, estimator='Natural', num_threads=1,
        max_sample_size=int(1e6)):
    """
    Calculate the angular two-point correlation function, :math:`w(\\theta)`.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` argument.

    For a step-by-step tutorial, see :ref:`galaxy_catalog_analysis_tutorial9`.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 2 numpy array containing ra,dec positions of points in degrees.

    theta_bins : array_like
        array of boundaries defining the angular distance bins in which pairs are
        counted.

    sample2 : array_like, optional
        Npts2 x 2 array containing ra,dec positions of points in degrees.

    randoms : array_like, optional
        Nran x 2 array containing ra,dec positions of points in degrees.  If no randoms
        are provided analytic randoms are used (only valid for for continious all-sky
        converage).

    do_auto : boolean, optional
        Boolean determines whether the auto-correlation function will
        be calculated and returned. Default is True.

    do_cross : boolean, optional
        Boolean determines whether the cross-correlation function will
        be calculated and returned. Only relevant when ``sample2`` is also provided.
        Default is True for the case where ``sample2`` is provided, otherwise False.

    estimator : string, optional
        Statistical estimator for the tpcf.
        Options are 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
        Default is ``Natural``.

    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed
        using the python ``multiprocessing`` module. Default is 1 for a purely serial
        calculation, in which case a multiprocessing Pool object will
        never be instantiated. A string 'max' may be used to indicate that
        the pair counters should use all available cores on the machine.

    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter.
        If sample size exeeds max_sample_size,
        the sample will be randomly down-sampled such that the subsample
        is equal to ``max_sample_size``. Default value is 1e6.

    Returns
    -------
    correlation_function(s) : numpy.array
        *len(theta_bins)-1* length array containing the correlation function
        :math:`w(\\theta)` computed in each of the bins defined by input ``theta_bins``.

        .. math::
            1 + w(\\theta) \\equiv \\mathrm{DD}(\\theta) / \\mathrm{RR}(\\theta),

        If ``estimator`` is set to 'Natural'.  :math:`\\mathrm{DD}(\\theta)` is the number
        of sample pairs with separations equal to :math:`\\theta`, calculated by the pair
        counter.  :math:`\\mathrm{RR}(\\theta)` is the number of random pairs with
        separations equal to :math:`\\theta`, and is counted internally using
        "analytic randoms" if ``randoms`` is set to None (see notes for an explanation),
        otherwise it is calculated using the pair counter.

        If ``sample2`` is passed as input
        (and if ``sample2`` is not exactly the same as ``sample1``),
        then three arrays of length *len(rbins)-1* are returned:

        .. math::
            w_{11}(\\theta), w_{12}(\\theta), w_{22}(\\theta),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and
        ``sample2``, and the autocorrelation of ``sample2``, respectively.
        If ``do_auto`` or ``do_cross`` is set to False,
        the appropriate sequence of results is returned.

    Notes
    -----
    Pairs are counted using `~halotools.mock_observables.npairs_3d`.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points on the
    sky:

    >>> from halotools.utils import sample_spherical_surface
    >>> Npts = 1000
    >>> angular_coords = sample_spherical_surface(Npts) #in degrees

    >>> theta_bins = np.logspace(-2,1,10)
    >>> w = angular_tpcf(angular_coords, theta_bins)

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial9`
    """

    #check input arguments using clustering helper functions
    function_args = (sample1, theta_bins, sample2, randoms, do_auto, do_cross,
        estimator, num_threads, max_sample_size)

    #pass arguments in, and get out processed arguments, plus some control flow variables
    sample1, theta_bins, sample2, randoms, do_auto, do_cross, num_threads,\
    _sample1_is_sample2 = _angular_tpcf_process_args(*function_args)

    #convert angular bins to coord lengths on a unit sphere
    chord_bins = chord_to_cartesian(theta_bins, radians=False)

    #convert samples and randoms to cartesian coordinates (x,y,z) on a unit sphere
    x, y, z = spherical_to_cartesian(sample1[:, 0], sample1[:, 1])
    sample1 = np.vstack((x, y, z)).T
    if _sample1_is_sample2:
        sample2 = sample1
    else:
        x, y, z = spherical_to_cartesian(sample2[:, 0], sample2[:, 1])
        sample2 = np.vstack((x, y, z)).T
    if randoms is not None:
        x, y, z = spherical_to_cartesian(randoms[:, 0], randoms[:, 1])
        randoms = np.vstack((x, y, z)).T

    def random_counts(sample1, sample2, randoms, chord_bins,
            num_threads, do_RR, do_DR, _sample1_is_sample2):
        """
        Count random pairs.
        """
        def area_spherical_cap(chord):
            """
            Calculate the area of a spherical cap on a unit sphere given the chord length
            """
            h = 1.0 - np.sqrt(1.0-chord**2)
            return np.pi*(chord**2+h**2)

        #randoms provided, so calculate random pair counts.
        if randoms is not None:
            if do_RR is True:
                RR = npairs_3d(randoms, randoms, chord_bins,
                            num_threads=num_threads)
                RR = np.diff(RR)
            else: RR=None
            if do_DR is True:
                D1R = npairs_3d(sample1, randoms, chord_bins,
                             num_threads=num_threads)
                D1R = np.diff(D1R)
            else: D1R=None
            if _sample1_is_sample2:
                D2R = None
            else:
                if do_DR is True:
                    D2R = npairs_3d(sample2, randoms, chord_bins,
                                 num_threads=num_threads)
                    D2R = np.diff(D2R)
                else: D2R=None
            return D1R, D2R, RR
        elif randoms is None:

            #set the number of randoms equal to the number of points in sample1
            NR = len(sample1)

            #do area calculations
            da = area_spherical_cap(chord_bins)
            da = np.diff(da)
            global_area = 4.0*np.pi  # surface area of a unit sphere

            #calculate randoms for sample1
            N1 = np.shape(sample1)[0]  # number of points in sample1
            rho1 = N1/global_area  # number density of points
            D1R = (N1)*(da*rho1)  # random counts are N**2*dv*rho

            N2 = np.shape(sample2)[0]
            rho2 = N2/global_area  # number density of points
            D2R = N2*(da*rho2)

            #calculate the random-random pairs.
            rhor = NR**2/global_area
            RR = (da*rhor)

            return D1R, D2R, RR

    def pair_counts(sample1, sample2, chord_bins,
            N_thread, do_auto, do_cross, _sample1_is_sample2):
        """
        Count data-data pairs.
        """

        if do_auto is True:
            D1D1 = npairs_3d(sample1, sample1, chord_bins, num_threads=num_threads)
            D1D1 = np.diff(D1D1)
        else:
            D1D1=None
            D2D2=None

        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross is True:
                D1D2 = npairs_3d(sample1, sample2, chord_bins, num_threads=num_threads)
                D1D2 = np.diff(D1D2)
            else: D1D2=None
            if do_auto is True:
                D2D2 = npairs_3d(sample2, sample2, chord_bins, num_threads=num_threads)
                D2D2 = np.diff(D2D2)
            else: D2D2=None

        return D1D1, D1D2, D2D2

    # What needs to be done?
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)
    if randoms is not None:
        NR = len(randoms)
    else:
        #set the number of randoms equal to the number of points in sample1
        #this is arbitrarily set, but must remain consistent!
        NR = N1

    #count data pairs
    D1D1, D1D2, D2D2 = pair_counts(sample1, sample2, chord_bins,
                                 num_threads, do_auto, do_cross, _sample1_is_sample2)
    #count random pairs
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, chord_bins,
                                 num_threads, do_RR, do_DR, _sample1_is_sample2)

    #check to see if any of the random counts contain 0 pairs.
    if D1R is not None:
        if np.any(D1R==0):
            msg = ("sample1 cross randoms has theta bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    if D2R is not None:
        if np.any(D2R==0):
            msg = ("sample2 cross randoms has theta bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)
    if RR is not None:
        if np.any(RR==0):
            msg = ("randoms cross randoms has theta bin(s) which contain no points. \n"
                   "Consider increasing the number of randoms, or using larger bins.")
            warn(msg)

    #run results through the estimator and return relavent/user specified results.
    if _sample1_is_sample2:
        xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)
        return xi_11
    else:
        if (do_auto is True) & (do_cross is True):
            xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)
            xi_12 = _TP_estimator(D1D2, D1R, RR, N1, N2, NR, NR, estimator)
            xi_22 = _TP_estimator(D2D2, D2R, RR, N2, N2, NR, NR, estimator)
            return xi_11, xi_12, xi_22
        elif (do_cross is True):
            xi_12 = _TP_estimator(D1D2, D1R, RR, N1, N2, NR, NR, estimator)
            return xi_12
        elif (do_auto is True):
            xi_11 = _TP_estimator(D1D1, D1R, D1R, N1, N1, NR, NR, estimator)
            xi_22 = _TP_estimator(D2D2, D2R, D2R, N2, N2, NR, NR, estimator)
            return xi_11, xi_22


def _angular_tpcf_process_args(sample1, theta_bins, sample2, randoms,
        do_auto, do_cross, estimator, num_threads, max_sample_size):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.angular_tpcf`.
    """

    sample1 = np.atleast_1d(sample1)

    if sample2 is not None:
        sample2 = np.atleast_1d(sample2)
        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("\n `sample1` and `sample2` are exactly the same, \n"
                   "only the auto-correlation will be returned.\n")
            warn(msg)
            do_cross=False
        else:
            _sample1_is_sample2 = False
    else:
        sample2 = sample1
        _sample1_is_sample2 = True

    if randoms is not None:
        randoms = np.atleast_1d(randoms)

    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0, len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            msg = ("\n `sample1` exceeds `max_sample_size` \n"
                   "downsampling `sample1`...")
            warn(msg)
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0, len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            msg = ("\n `sample1` exceeds `max_sample_size` \n"
                   "downsampling `sample1`...")
            warn(msg)
        if len(sample2) > max_sample_size:
            inds = np.arange(0, len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            msg = ("\n `sample2` exceeds `max_sample_size` \n"
                    "downsampling `sample2`...")
            warn(msg)

    theta_bins = np.atleast_1d(theta_bins)
    theta_max = np.max(theta_bins)
    try:
        assert theta_bins.ndim == 1
        assert len(theta_bins) > 1
        if len(theta_bins) > 2:
            assert array_is_monotonic(theta_bins, strict=True) == 1
    except AssertionError:
        msg = ("\n Input `theta_bins` must be a monotonically increasing 1-D \n"
               "array with at least two entries.")
        raise HalotoolsError(msg)

    #check for input parameter consistency
    if (theta_max >= 180.0):
        msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than 180.0 deg. \n")
        raise HalotoolsError(msg)

    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\n `sample1` and `sample2` must have same dimension.\n')
        raise HalotoolsError(msg)

    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('\n `do_auto` and `do_cross` keywords must be of type boolean.')
        raise HalotoolsError(msg)

    num_threads = get_num_threads(num_threads)

    verify_tpcf_estimator(estimator)

    return sample1, theta_bins, sample2, randoms, do_auto, do_cross, num_threads,\
           _sample1_is_sample2
