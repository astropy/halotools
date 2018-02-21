r"""
Module containing the `~halotools.mock_observables.alignments.gi_plus_projected` function used to
calculate the projected gravitational shear-intrinsic ellipticity (GI) correlation
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import pi

from .alignment_helpers import process_projected_alignment_args
from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_separation_bins_array, get_line_of_sight_bins_array, get_period, get_num_threads)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..pair_counters import positional_marked_npairs_xy_z, marked_npairs_xy_z

__all__ = ['gi_plus_projected']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def gi_plus_projected(sample1, orientations1, ellipticities1, sample2, rp_bins, pi_max,
            randoms1=None, randoms2=None, weights1=None, weights2=None,
            ran_weights1=None, ran_weights2=None, estimator='Landy-Szalay',
            period=None, num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the projected gravitational shear-intrinsic ellipticity correlation function (GI),
    :math:`w_{g+}(r_p)`, where :math:`r_p` is the separation perpendicular to the line-of-sight (LOS)
    between two galaxies.  See the 'Notes' section for details of this calculation.

    The first two dimensions define the plane for perpendicular distances.  The third
    dimension is used for parallel distances, i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate. This is the 'distant observer' approximation.

    Note in particular that the `~halotools.mock_observables.alignments.gi_plus_projected` function does not
    accept angular coordinates for the input ``sample1`` or ``sample2``.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points with associated
        orientations and ellipticities.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    orientations1 : array_like
        Npts1 x 2 numpy array containing projected orientation vectors for each point in ``sample1``.
        these will be normalized if not already.

    ellipticities1: array_like
        Npts1 x 1 numpy array containing ellipticities for each point in ``sample1``.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.

    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    pi_max : float
        maximum LOS distance defining the projection integral length-scale in the z-dimension.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    randoms1 : array_like, optional
        Nran1 x 3 array containing 3-D positions of randomly distributed points corresponding to ``sample1``.
        If no randoms are provided (the default option), the
        calculation can proceed using analytical randoms
        (only valid for periodic boundary conditions).

    randoms2 : array_like, optional
        Nran2 x 3 array containing 3-D positions of randomly distributed points corresponding to ``sample2``.
        If no randoms are provided (the default option), the
        calculation can proceed using analytical randoms
        (only valid for periodic boundary conditions).

    weights1 : array_like, optional
        Npts1 array of weghts.  If this parameter is not specified, it is set to numpy.ones(Npts1).

    weights2 : array_like, optional
        Npts2 array of weghts.  If this parameter is not specified, it is set to numpy.ones(Npts2).

    ran_weights1 : array_like, optional
        Npran1 array of weghts.  If this parameter is not specified, it is set to numpy.ones(Nran1).

    ran_weights2 : array_like, optional
        Nran2 array of weghts.  If this parameter is not specified, it is set to numpy.ones(Nran2).

    estimator :  string, optional
        string indicating which estimator to use

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity,
        in which case ``randoms`` must be provided.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed
        using the python ``multiprocessing`` module. Default is 1 for a purely serial
        calculation, in which case a multiprocessing Pool object will
        never be instantiated. A string 'max' may be used to indicate that
        the pair counters should use all available cores on the machine.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by how points
        will be apportioned into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use Lbox/10 in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.

    approx_cell2_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for
        ``approx_cell1_size`` for details.

    Returns
    -------
    correlation_function : numpy.array
        *len(rp_bins)-1* length array containing the correlation function :math:`w_{g+}(r_p)`
        computed in each of the bins defined by input ``rp_bins``.

    Notes
    -----

    The projected GI-correlation function is calculated as:

    .. math::
        w_{g+}(r_p) = 2 \int_0^{\pi_{\rm max}} \xi_{g+}(r_p, \pi) \mathrm{d}\pi

    If the Landy-Szalay estimator is indicated, the correlation function is estimated as:

    .. math::
        \xi_{g+}(r_p, \pi) = \frac{S_{+}D-S_{+}R}{R_sR}

     where

    .. math::
        S_{+}D = \sum_{i \neq j} w_jw_i e_{+}(j|i)

    :math:`w_j` and :math:`w_j` are weights.  Weights are set to 1 for all galaxies by default.
    The alingment of the :math:`j`-th galaxy relative to the direction to the :math:`i`-th galaxy is given by:

    .. math::
        e_{+}(j|i) = e_j\cos(2\phi)

    where :math:`e_j` is the ellipticity of the :math:`j`-th galaxy.  :math:`\phi` is the angle between the
    orientation vector, :math:`\vec{o}_j`, and the projected direction between the :math:`j`-th
    and :math:`i`-th galaxy, :math:`\vec{r}_{p i,j}`.

    .. math::
        \cos(\phi) = \vec{o}_j \cdot \vec{r}_{p i,j}

    :math:`S_{+}R` is analgous to :math:`S_{+}D` but instead is computed
    with respect to a "random" catalog of galaxies.  :math:`R_sR` are random pair counts,
    where :math:`R_s` corresponds to the shapes sample, i.e. the sample with orienttions
    and ellipticies, ``sample1``, and R correspoinds to ``sample2``.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic cube of Lbox = 250 Mpc/h.

    >>> Npts = 1000
    >>> Lbox = 250

    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack((x,y,z)).T

    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array`
    convenience function for this same purpose, which provides additional wrapper
    behavior around `numpy.vstack` such as placing points into redshift-space.

    We then create a set of random orientation vectors and ellipticities for each point

    >>> random_orientations = np.random.random((Npts,2))
    >>> random_ellipticities = np.random.random(Npts)

    We can the calculate the projected auto-GI correlation between these points:

    >>> rp_bins = np.logspace(-1,1,10)
    >>> pi_max = 0.25
    >>> w = gi_plus_projected(sample1, random_orientations, random_ellipticities, sample1, rp_bins, pi_max, period=Lbox)

    """

    # process arguments
    alignment_args = (sample1, orientations1, ellipticities1, weights1,
                      sample2, None, None, weights2,
                      randoms1, ran_weights1, randoms2, ran_weights2)
    sample1, orientations1, ellipticities1, weights1, sample2,\
        orientations2, ellipticities2, weights2, randoms1, ran_weights1,\
        randoms2, ran_weights2 = process_projected_alignment_args(*alignment_args)

    function_args = (sample1, rp_bins, pi_max, sample2, randoms1, randoms2,
        period, num_threads, approx_cell1_size, approx_cell2_size)
    sample1, rp_bins, pi_bins, sample2, randoms1, randoms2,\
        period, num_threads, PBCs, no_randoms = _gi_plus_projected_process_args(*function_args)

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)
    if no_randoms:  # set random density the the same as the sampels
        NR1 = N1
        NR2 = N2
    else:
        NR1 = len(randoms1)
        NR2 = len(randoms2)

    #define merk vectors to use in pair counting
    # sample 1
    marks1 = np.ones((N1, 3))
    marks1[:, 0] = ellipticities1 * weights1
    marks1[:, 1] = orientations1[:, 0]
    marks1[:, 2] = orientations1[:, 1]
    # sample 2
    marks2 = weights2
    # randoms 1
    ran_marks1 = np.ones((NR1, 3))
    ran_marks1[:, 0] = ran_weights1
    ran_marks1[:, 1] = 0  # dummy
    ran_marks1[:, 2] = 0  # dummy
    # randoms 2
    ran_marks2 = ran_weights2

    # define pi bins
    pi_bins = np.array([0.0, pi_max])

    do_SD, do_SR, do_RR = GI_estimator_requirements(estimator)

    # count marked pairs
    if do_SD:
        SD = marked_pair_counts(sample1, sample2,  marks1,  marks2,
                                rp_bins, pi_bins, period, num_threads,
                                approx_cell1_size, approx_cell2_size)
    else:
        SD = None

    # count marked random pairs
    if do_SR:
        if no_randoms:
            SR = 0.0
        else:
            SR = marked_pair_counts(sample1, randoms2, marks1, ran_marks2,
                                    rp_bins, pi_bins, period, num_threads,
                                    approx_cell1_size, approx_cell2_size)
    else:
        SR = None

    # count random pairs
    if do_RR:
        RR = random_counts(randoms1, randoms2, ran_weights1, ran_weights2,
                           rp_bins, pi_bins, N1, N2, no_randoms, period, PBCs,
                           num_threads, approx_cell1_size, approx_cell2_size)
    else:
        RR = None

    result = GI_estimator(SD, SR, RR, N1, N2, NR1, NR2, estimator)

    return result*2.0*pi_max  # factor of 2pi_max accounts for integration


def GI_estimator(SD, SR, RR, N1, N2, NR1, NR2, estimator='Landy-Szalay'):
    r"""
    apply the supplied GI estimator to calculate the correlation function.
    """

    if estimator == 'Landy-Szalay':
        factor = (NR1*NR2)/(N1*N2)
        return factor*(SD-SR)/RR
    else:
        msg = ('The estimator provided is not supported.')
        raise ValueError(msg)


def GI_estimator_requirements(estimator):
    r"""
    Return the requirments for the supplied GI estimator.
    """
    do_SD = False
    do_SR = False
    do_RR = False

    if estimator == 'Landy-Szalay':
        do_SD = True
        do_SR = True
        do_RR = True
        return do_SD, do_SR, do_RR
    else:
        msg = ('The estimator provided is not supported.')
        raise ValueError(msg)


def marked_pair_counts(sample1, sample2, weights1, weights2, rp_bins, pi_bins, period,
        num_threads, approx_cell1_size, approx_cell2_size):
    r"""
    Count marked pairs.
    """

    weight_func_id = 2
    SD = positional_marked_npairs_xy_z(sample1, sample2, rp_bins, pi_bins, period=period,
        weights1=weights1, weights2=weights2, weight_func_id=weight_func_id,
        num_threads=num_threads, approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell1_size)[0]
    SD = np.diff(np.diff(SD, axis=0), axis=1)
    SD = SD.flatten()

    return SD


def random_counts(randoms1, randoms2, ran_weights1, ran_weights2, rp_bins, pi_bins,
                  N1, N2, no_randoms, period,
                  PBCs, num_threads, approx_cell1_size, approx_cell2_size):
    r"""
    Count random pairs.
    """

    if no_randoms is False:
        RR = marked_npairs_xy_z(randoms1, randoms2, rp_bins, pi_bins,
                period=period, num_threads=num_threads, weight_func_id=1,
                weights1=ran_weights1, weights2=ran_weights2,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell2_size)
        RR = np.diff(np.diff(RR, axis=0), axis=1)
        RR = RR.flatten()

        return RR
    else:
        # set 'number' or randoms
        # setting Nran to Ndata makes normalization simple
        NR1 = N1
        NR2 = N2

        # do volume calculations
        v = cylinder_volume(rp_bins, 2.0*pi_bins)
        dv = np.diff(np.diff(v, axis=0), axis=1)
        global_volume = period.prod()

        # calculate the expected random-random pairs.
        rhor = (NR1*NR2)/global_volume
        RR = (dv*rhor)

        return RR.flatten()


def cylinder_volume(R, h):
    r"""
    Calculate the volume of a cylinder(s), used for the analytical randoms.
    """
    return pi*np.outer(R**2.0, h)


def _gi_plus_projected_process_args(sample1, rp_bins, pi_max, sample2, randoms1, randoms2,
        period, num_threads, approx_cell1_size, approx_cell2_size):
    r"""
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.alignments.w_gplus`.
    """
    sample1 = enforce_sample_has_correct_shape(sample1)

    if randoms1 is not None:
        randoms1 = np.atleast_1d(randoms1)
        no_randoms1 = False
    else: no_randoms1 = True

    if randoms2 is not None:
        randoms2 = np.atleast_1d(randoms2)
        no_randoms2 = False
    else: no_randoms2 = True

    #if one of the randoms is missing, raise an error
    no_randoms = True
    if no_randoms1:
        if no_randoms2 is False:
            msg = "if one set of randoms is provided, both randoms must be provided.\n"
            raise ValueError(msg)
    elif no_randoms2:
        if no_randoms1 is False:
            msg = "if one set of randoms is provided, both randoms must be provided.\n"
            raise ValueError(msg)
    else:
        no_randoms = False

    pi_max = float(pi_max)
    pi_bins = np.array([0.0, pi_max])

    rp_bins = get_separation_bins_array(rp_bins)
    rp_max = np.amax(rp_bins)

    period, PBCs = get_period(period)

    _enforce_maximum_search_length([rp_max, rp_max, pi_max], period)

    if (randoms1 is None) & (PBCs is False):
        msg = "If no PBCs are specified, both randoms must be provided.\n"
        raise ValueError(msg)

    num_threads = get_num_threads(num_threads)

    return sample1, rp_bins, pi_bins, sample2, randoms1, randoms2, period, num_threads, PBCs, no_randoms
