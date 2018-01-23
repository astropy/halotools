r"""
Module containing the `~halotools.mock_observables.w_gplus` function used to
calculate the projected gravitational shear-intrinsic ellipticity correlation
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .alignment_helpers import process_projected_alignment_args
from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_separation_bins_array, get_line_of_sight_bins_array, get_period, get_num_threads)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..pair_counters import positional_marked_npairs_xy_z

__all__ = ['w_gplus']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def w_gplus(sample1, orientations1, ellipticities1, sample2, rp_bins, pi_max,
            randoms=None, period=None, num_threads=1,
            approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the projected projected gravitational shear-intrinsic ellipticity correlation function,
    :math:`w_{g+}(r_p)`, where :math:`r_p` is the separation perpendicular to the line-of-sight (LOS).

    The first two dimensions define the plane for perpendicular distances.  The third
    dimension is used for parallel distances, i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate. This is the 'distant observer' approximation.

    Note in particular that the `~halotools.mock_observables.w_gplus` function does not
    accept angular coordinates for the input ``sample1`` or ``sample2``.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points with associated orientations.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    orientations1 : array_like
        Npts1 x 2 numpy array containing projected orientation vectors for each point in ``sample1``.

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
    
    randoms : array_like, optional
        Nran x 3 array containing 3-D positions of randomly distributed points.
        If no randoms are provided (the default option), the
        calculation can proceed using analytical randoms
        (only valid for periodic boundary conditions).

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

    .. math::
        w_{g+}(r_p) = \frac{S_{+}D}{RR}

    where

    .. math::
        S_{+}D = \sum_{i \neq j}w_j e_{+}(j|i)

    and the alingment of the :math:`j`-th galaxy relative the direction of the :math:`i`-th galaxy is given by:

    .. math::
        e_{+}(j|i) = e\cos(2\phi)
    
    where :math:`\phi` is the angle between the orientation vector and the vector connecting the projected positions.

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

    We then create a set of random orientation vectors and ellipticities for each point

    >>> random_orientations = np.random.random((len(data),2))
    >>> random_ellipticities = np.random.random((len(data))

    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array`
    convenience function for this same purpose, which provides additional wrapper
    behavior around `numpy.vstack` such as placing points into redshift-space.

    >>> rp_bins = np.logspace(-1,1,10)
    >>> pi_max = 0.25
    >>> w = w_gplus(sample1, random_orientations, random_ellipticities, rp_bins, pi_max, period=Lbox)

    """

    function_args = (sample1, orientations1, ellipticities1, sample2, None, None,
        rp_bins, pi_max, period, num_threads, approx_cell1_size, approx_cell2_size)

    sample1, orientations1, ellipticities1, sample2, None, None, rp_bins, pi_max, period,
    num_threads, approx_cell1_size, approx_cell2_size, PBCs = process_projected_alignment_args(*function_args)

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)
    
    #define weights to use in pair counting
    weights1 = np.ones((N2,3))
    weights1[:,0] = ellipticities1
    weights1[:,1] = orientations1[:0]
    weights1[:,2] = orientations1[:1]
    weights2 = np.ones((N1,3)) #just a dummy, the orientations arent used for this sample 

    # count marked pairs
    SD = marked_pair_counts(sample1, sample2,  weights1,  weights2,
        rp_bins, pi_max, period, num_threads, approx_cell1_size, approx_cell2_size)
    
    # count marked random pairs
    if do_SR:
    	if randoms is not None:
    		ran_weights = weights2
    		SR = marked_pair_counts(sample1, randoms, weights1, ran_weights,
        rp_bins, pi_max, period, num_threads, do_auto, do_cross,
        _sample1_is_sample2, approx_cell1_size, approx_cell2_size)
    	else:
    		SR = 

    # count random pairs
    RR = random_counts()






def marked_pair_counts(sample1, sample2, weights1, weights2, rp_bins, pi_max, period,
        num_threads, approx_cell1_size, approx_cell2_size):
    """
    Count data pairs.
    """

    weight_func_id = 1
    SD = positional_marked_npairs_xy_z(sample1, sample2, rp_bins, pi_max, period=period,
        weights1=weights1, weights2=weights1, weight_func_id=weight_func_id,
        num_threads=num_threads, approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell1_size)
    SD = np.diff(np.diff(SD, axis=0), axis=1)

    return SD


def random_counts(sample1, sample2, randoms, rp_bins, pi_bins, period,
        PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size, approx_cellran_size):
    r"""
    Count random pairs.
    """

    # No PBCs, randoms must have been provided.
    if randoms is not None:
        if do_RR is True:
            RR = npairs_xy_z(randoms, randoms, rp_bins, pi_bins,
                period=period, num_threads=num_threads,
                approx_cell1_size=approx_cellran_size,
                approx_cell2_size=approx_cellran_size)
            RR = np.diff(np.diff(RR, axis=0), axis=1)
        else:
            RR = None
        if do_DR is True:
            D1R = npairs_xy_z(sample1, randoms, rp_bins, pi_bins,
                period=period, num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cellran_size)
            D1R = np.diff(np.diff(D1R, axis=0), axis=1)
        else:
            D1R = None
        if _sample1_is_sample2:  # calculating the cross-correlation
            D2R = None
        else:
            if do_DR is True:
                D2R = npairs_xy_z(sample2, randoms, rp_bins, pi_bins,
                    period=period, num_threads=num_threads,
                    approx_cell1_size=approx_cell2_size,
                    approx_cell2_size=approx_cellran_size)
                D2R = np.diff(np.diff(D2R, axis=0), axis=1)
            else:
                D2R = None

        return D1R, D2R, RR
    # PBCs and no randoms--calculate randoms analytically.
    elif randoms is None:

        # set the number of randoms equal to the number of points in sample1
        NR = len(sample1)

        # do volume calculations
        v = cylinder_volume(rp_bins, 2.0*pi_bins)  # volume of spheres
        dv = np.diff(np.diff(v, axis=0), axis=1)  # volume of annuli
        global_volume = period.prod()

        # calculate randoms for sample1
        N1 = np.shape(sample1)[0]
        rho1 = N1/global_volume
        D1R = (N1)*(dv*rho1)  # read note about pair counter

        # calculate randoms for sample2
        N2 = np.shape(sample2)[0]
        rho2 = N2/global_volume
        D2R = N2*(dv*rho2)  # read note about pair counter

        # calculate the random-random pairs.
        rhor = NR**2/global_volume
        RR = (dv*rhor)  # RR is only the RR for the cross-correlation.

        return D1R, D2R, RR




