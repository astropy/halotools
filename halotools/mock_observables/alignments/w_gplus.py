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


def w_gplus(sample1, ellipticities1, sample2, alignments2, ellipticities2, rp_bins, pi_max,
            period=None, num_threads=1,
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
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

   ellipticities1: array_like

   sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function. Default is None, in which case only the
        auto-correlation function will be calculated.

    alignments2 : array_like

    ellipticities2: array_like

    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    pi_max : float
        maximum LOS distance defining the projection integral length-scale in the z-dimension.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

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
    correlation_function(s) : numpy.array
        *len(rp_bins)-1* length array containing the correlation function :math:`w_{g+}(r_p)`
        computed in each of the bins defined by input ``rp_bins``.

        If ``sample2`` is not None (and not exactly the same as ``sample1``),
        three arrays of length *len(rp_bins)-1* are returned:

        .. math::
            w_{g+11}(r_p), \ w_{g+12}(r_p), \ w_{g+22}(r_p),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1``
        and ``sample2``, and the autocorrelation of ``sample2``.  If ``do_auto`` or ``do_cross``
        is set to False, the appropriate result(s) is not returned.

    Notes
    -----

    .. math::
        w_{g+}(r_p) = \frac{S_{+}D}{RR}

    where

    .. math::
        S_{+}D = \sum_{i \neq j}w_j e_{+}(j|i)

    and the alingment of the :math:`j`-th galaxy relative the direction of the :math:`i`-th galaxy is given by:

    .. math::
        e_{+}(j|i) = -e_1\cos(2\phi) - e_2\sin(2\phi)

    """

    function_args = (sample1, sample2, None, ellipticities1, alignments2, ellipticities2,
        rp_bins, pi_max, period, num_threads, approx_cell1_size, approx_cell2_size)

    sample1, alignments1, rp_bins, pi_max, sample2, alignments2, period, do_auto, do_cross, num_threads,\
        _sample1_is_sample2, PBCs = process_projected_alignment_args(*function_args)

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)

    # count marked pairs
    D1D1, D1D2, D2D2 = marked_pair_counts(sample1, sample2, alignments1, alignments2,
        rp_bins, pi_max, period, num_threads, do_auto, do_cross,
        _sample1_is_sample2, approx_cell1_size, approx_cell2_size)




def marked_pair_counts(sample1, sample2, alignments1, alignments2, rp_bins, pi_max, period,
        num_threads, do_auto, do_cross, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size):
    """
    Count data pairs.
    """
    D1D1 = positional_marked_npairs_xy_z(sample1, sample1, rp_bins, pi_max, period=period,
        weights1=alignments1, weights2=alignments1,
        num_threads=num_threads, approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell1_size)
    D1D1 = np.diff(np.diff(D1D1, axis=0), axis=1)
    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = positional_marked_npairs_xy_z(sample1, sample2, rp_bins, pi_max, period=period,
                weights1=alignments1, weights2=alignments2,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell2_size)
            D1D2 = np.diff(np.diff(D1D2, axis=0), axis=1)
        else:
            D1D2 = None
        if do_auto is True:
            D2D2 = positional_marked_npairs_xy_z(sample2, sample2, rp_bins, pi_max, period=period,
                weights1=alignments2, weights2=alignments2,
                num_threads=num_threads,
                approx_cell1_size=approx_cell2_size,
                approx_cell2_size=approx_cell2_size)
            D2D2 = np.diff(np.diff(D2D2, axis=0), axis=1)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2




