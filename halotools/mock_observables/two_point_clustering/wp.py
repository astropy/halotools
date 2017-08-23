r"""
Module containing the `~halotools.mock_observables.wp` function used to
calculate the projected two-point correlation function (aka projected galaxy clustering).
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .rp_pi_tpcf import rp_pi_tpcf, _rp_pi_tpcf_process_args


__all__ = ['wp']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def wp(sample1, rp_bins, pi_max, sample2=None, randoms=None, period=None,
        do_auto=True, do_cross=True, estimator='Natural', num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None,
        approx_cellran_size=None, seed=None):
    r"""
    Calculate the projected two point correlation function, :math:`w_{p}(r_p)`,
    where :math:`r_p` is the separation perpendicular to the line-of-sight (LOS).

    The first two dimensions define the plane for perpendicular distances.  The third
    dimension is used for parallel distances, i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate. This is the 'distant observer' approximation.

    Note in particular that the `~halotools.mock_observables.wp` function does not
    accept angular coordinates for the input ``sample1`` or ``sample2``. If you
    are trying to calculate projected galaxy clustering from a set of observational data,
    the `~halotools.mock_observables.wp` function is not what you want.
    To perform such a calculation, refer to the appropriate function of the Corrfunc code
    written by Manodeep Sinha, available at https://github.com/manodeep/Corrfunc,
    which *can* be used to calculate projected clustering from a set of observational data.

    Example calls to the `~halotools.mock_observables.wp` function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.

    See also :ref:`galaxy_catalog_analysis_tutorial4` for a step-by-step tutorial on
    how to use this function on a mock galaxy catalog.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    pi_max : float
        maximum LOS distance defining the projection integral length-scale in the z-dimension.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function. Default is None, in which case only the
        auto-correlation function will be calculated.

    randoms : array_like, optional
        Nran x 3 array containing 3-D positions of randomly distributed points.
        If no randoms are provided (the default option),
        calculation of the tpcf can proceed using analytical randoms
        (only valid for periodic boundary conditions).

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity,
        in which case ``randoms`` must be provided.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

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

    approx_cellran_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for randoms.  See comments for
        ``approx_cell1_size`` for details.

    seed : int, optional
        Random number seed used to randomly downsample data, if applicable.
        Default is None, in which case downsampling will be stochastic.

    Returns
    -------
    correlation_function(s) : numpy.array
        *len(rp_bins)-1* length array containing the correlation function :math:`w_p(r_p)`
        computed in each of the bins defined by input ``rp_bins``.

        If ``sample2`` is not None (and not exactly the same as ``sample1``),
        three arrays of length *len(rp_bins)-1* are returned:

        .. math::
            w_{p11}(r_p), \ w_{p12}(r_p), \ w_{p22}(r_p),

        the autocorrelation of ``sample1``, the cross-correlation between ``sample1``
        and ``sample2``, and the autocorrelation of ``sample2``.  If ``do_auto`` or ``do_cross``
        is set to False, the appropriate result(s) is not returned.

    Notes
    -----
    The projected correlation function is calculated by integrating the
    redshift space two point correlation function using
    `~halotools.mock_observables.rp_pi_tpcf`:

    .. math::
        w_p(r_p) = \int_0^{\pi_{\rm max}}2.0\xi(r_p,\pi)\mathrm{d}\pi

    where :math:`\pi_{\rm max}` is ``pi_max`` and :math:`\xi(r_p,\pi)`
    is the redshift space correlation function.

    For a higher-performance implementation of the wp function,
    see the Corrfunc code written by Manodeep Sinha, available at
    https://github.com/manodeep/Corrfunc.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic cube with Lbox = 250 Mpc/h.

    >>> Npts = 1000
    >>> Lbox = 250.

    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> coords = np.vstack((x,y,z)).T

    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array`
    convenience function for this same purpose, which provides additional wrapper
    behavior around `numpy.vstack` such as placing points into redshift-space.

    >>> rp_bins = np.logspace(-1,1,10)
    >>> pi_max = 10
    >>> xi = wp(coords, rp_bins, pi_max, period=Lbox)

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial4`

    """

    # define the volume to search for pairs
    pi_max = float(pi_max)
    pi_bins = np.array([0.0, pi_max])

    # process input parameters
    function_args = (sample1, rp_bins, pi_bins, sample2, randoms, period, do_auto,
        do_cross, estimator, num_threads,
        approx_cell1_size, approx_cell2_size, approx_cellran_size, seed)
    sample1, rp_bins, pi_bins, sample2, randoms, period, do_auto, do_cross, num_threads,\
        _sample1_is_sample2, PBCs = _rp_pi_tpcf_process_args(*function_args)

    if _sample1_is_sample2:
        sample2 = None

    # pass the arguments into the redshift space TPCF function
    result = rp_pi_tpcf(sample1, rp_bins=rp_bins, pi_bins=pi_bins,
        sample2=sample2, randoms=randoms,
        period=period, do_auto=do_auto, do_cross=do_cross,
        estimator=estimator, num_threads=num_threads,
        approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell2_size,
        approx_cellran_size=approx_cellran_size)

    # return the results.
    if _sample1_is_sample2:
        D1D1 = result[:, 0]
        wp_D1D1 = 2.0*D1D1*pi_max
        return wp_D1D1
    else:
        if (do_auto is True) & (do_cross is True):
            D1D1 = result[0][:, 0]
            D1D2 = result[1][:, 0]
            D2D2 = result[2][:, 0]
            wp_D1D1 = 2.0*D1D1*pi_max
            wp_D1D2 = 2.0*D1D2*pi_max
            wp_D2D2 = 2.0*D2D2*pi_max
            return wp_D1D1, wp_D1D2, wp_D2D2
        elif (do_auto is True) & (do_cross is False):
            D1D1 = result[0][:, 0]
            D2D2 = result[1][:, 0]
            wp_D1D1 = 2.0*D1D1*pi_max
            wp_D2D2 = 2.0*D2D2*pi_max
            return wp_D1D1, wp_D2D2
        elif (do_auto is False) & (do_cross is True):
            D1D2 = result[:, 0]
            wp_D1D2 = 2.0*D1D2*pi_max
            return wp_D1D2
