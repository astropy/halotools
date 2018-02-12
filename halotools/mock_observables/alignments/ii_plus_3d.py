r"""
Module containing the `~halotools.mock_observables.alignments.ii_plus_3d` function used to
calculate the projected intrinsic ellipticity-ellipticity (II) correlation
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from math import pi

from .alignment_helpers import process_3d_alignment_args
from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_separation_bins_array, get_line_of_sight_bins_array, get_period, get_num_threads)
from ..pair_counters.mesh_helpers import _enforce_maximum_search_length
from ..pair_counters import positional_marked_npairs_3d, marked_npairs_3d

__all__ = ['ii_plus_3d']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def ii_plus_3d(sample1, orientations1, ellipticities1, sample2, orientations2, ellipticities2,
            rp_bins, pi_max, randoms1=None, randoms2=None, weights1=None, weights2=None,
            ran_weights1=None, ran_weights2=None, estimator='Natural',
            period=None, num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the intrinsic ellipticity-ellipticity correlation function (II),
    :math:`w_{++}(r)`.  See the 'Notes' section for details of this calculation.

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
        Npts2 x 3 array containing 3-D positions of points with associated
        orientations and ellipticities.

    orientations12 : array_like
        Npts1 x 2 numpy array containing projected orientation vectors for each point in ``sample2``.
        these will be normalized if not already.

    ellipticities2: array_like
        Npts1 x 1 numpy array containing ellipticities for each point in ``sample2``.

    rbins : array_like
        array of boundaries defining the radial bins in which
        pairs are counted.
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
        *len(rbins)-1* length array containing the correlation function :math:`w_{g+}(r)`
        computed in each of the bins defined by input ``rbins``.

    Notes
    -----

    The II-correlation function is calculated as:

    .. math::
        w_{++}(r) = 2 \int_0^{\pi_{\rm max}} \xi_{++}(r_p, \pi) \mathrm{d}\pi


    If the Natural estimator is indicated, the projected II-correlation function is calculated as:

    .. math::
        \xi_{++}(r) = \frac{S_{+}S_{+}}{R_sR_s}

    where

    .. math::
        S_{+}S_{+} = \sum_{i \neq j} w_jw_i e_{+}(j|i)e_{+}(i|j)

    :math:`w_j` and :math:`w_j` are weights.  Weights are set to 1 for all galaxies by default.
    The alingment of the :math:`j`-th galaxy relative to the direction to the :math:`i`-th galaxy is given by:

    .. math::
        e_{+}(j|i) = e_j\cos(2\phi)

    where :math:`e_j` is the ellipticity of the :math:`j`-th galaxy.  :math:`\phi` is the angle between the
    orientation vector, :math:`\vec{o}_j`, and the direction between the :math:`j`-th
    and :math:`i`-th galaxy, :math:`\vec{r}_{i,j}`.

    .. math::
        \cos(\phi) = \vec{o}_j \cdot \vec{r}_{i,j}

    :math:`R_sR_s` are random pair counts,

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

    >>> random_orientations = np.random.random((Npts,3))
    >>> random_ellipticities = np.random.random(Npts)

    We can the calculate the projected auto-GI correlation between these points:

    >>> rbins = np.logspace(-1,1,10)
    >>> w = ii_plus_3d(sample1, random_orientations, random_ellipticities, sample1, random_orientations, random_ellipticities, rbins, period=Lbox)

    """