"""
Module containing functions used to determine various ways of quantifying
the large-scale density of a set of points.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ..pair_counters import npairs_per_object_3d

from ...custom_exceptions import HalotoolsError


__all__ = ('large_scale_density_spherical_volume', )

__author__ = ('Andrew Hearin', )

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


def large_scale_density_spherical_volume(sample, tracers, radius,
        period=None, sample_volume=None, num_threads=1, approx_cell1_size=None,
        norm_by_mean_density=False):
    """
    Calculate the mean density of the input ``sample``
    from an input set of tracer particles using a sphere centered on each point
    in the input ``sample`` as the tracer volume.

    Around each point in the input ``sample``, a sphere of the input ``radius``
    is placed and the number of points in the input ``tracers`` is counted,
    optionally accounting for box periodicity.
    The `large_scale_density_spherical_volume` function returns the mean number density
    of tracer particles in each such sphere, optionally normalizing this result
    by the global mean number density of tracer particles in the entire sample volume.

    Parameters
    ------------
    sample : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample`` and ``tracers`` arguments.

    tracers : array_like
        Npts2 x 3 numpy array containing 3-D positions of tracers.

    radius : float
        Radius of the sphere used to define the volume inside which the
        number density of tracers is calculated.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity,
        and an input ``sample_volume`` must be provided.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample_volume : float, optional
        If period is set to None, you must specify the effective volume of the sample.
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

    norm_by_mean_density : bool, optional
        If set to True, the returned number density will be normalized by
        the global number density of tracer particles averaged across the
        entire ``sample_volume``. Default is False.

    Returns
    --------
    number_density : array_like
        Length-Npts1 array of number densities

    Examples
    ---------
    >>> npts1, npts2 = 100, 200
    >>> sample = np.random.random((npts1, 3))
    >>> tracers = np.random.random((npts2, 3))
    >>> radius = 0.1
    >>> result = large_scale_density_spherical_volume(sample, tracers, radius, period=1)

    """
    sample, tracers, rbins, period, sample_volume, num_threads, approx_cell1_size = (
        _large_scale_density_spherical_volume_process_args(
            sample, tracers, radius, period, sample_volume, num_threads, approx_cell1_size)
        )

    result = npairs_per_object_3d(sample, tracers, rbins, period=period,
        num_threads=num_threads, approx_cell1_size=approx_cell1_size)[:, 0]

    environment_volume = (4/3.)*np.pi*radius**3
    number_density = result/environment_volume

    if norm_by_mean_density is True:
        mean_rho = tracers.shape[0]/sample_volume
        return number_density/mean_rho
    else:
        return number_density


def _large_scale_density_spherical_volume_process_args(
        sample, tracers, radius, period, sample_volume, num_threads, approx_cell1_size):
    """
    """
    sample = np.atleast_1d(sample)
    tracers = np.atleast_1d(tracers)
    rbins = np.atleast_1d(radius).astype(float)
    rbins = np.append(rbins, rbins[0]+0.0001)

    if period is None:
        if sample_volume is None:
            msg = ("If period is None, you must pass in ``sample_volume``.")
            raise HalotoolsError(msg)
        else:
            sample_volume = float(sample_volume)
    else:
        period = np.atleast_1d(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = ("\nInput ``period`` must either be a float or length-3 sequence")
            raise HalotoolsError(msg)
        if sample_volume is None:
            sample_volume = period.prod()
        else:
            msg = ("If period is not None, do not pass in sample_volume")
            raise HalotoolsError(msg)

    return sample, tracers, rbins, period, sample_volume, num_threads, approx_cell1_size
