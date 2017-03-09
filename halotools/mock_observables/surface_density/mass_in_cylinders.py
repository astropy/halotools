"""
Module containing the `~halotools.mock_observables.surface_density_in_annulus`
and `~halotools.mock_observables.surface_density_in_cylinder` functions used to
calculate galaxy-galaxy lensing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .weighted_npairs_xy import weighted_npairs_xy
from .weighted_npairs_per_object_xy import weighted_npairs_per_object_xy

from ..mock_observables_helpers import (get_num_threads, get_separation_bins_array,
    get_period, enforce_sample_respects_pbcs, enforce_sample_has_correct_shape)


__all__ = ('total_mass_enclosed_in_stack_of_cylinders', 'total_mass_enclosed_per_cylinder')
__author__ = ('Andrew Hearin', )


def total_mass_enclosed_in_stack_of_cylinders(centers, particles,
        particle_masses, downsampling_factor, rp_bins, period,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r""" Calculate the total mass enclosed by a stack of cylinders of infinite length.

    Parameters
    ----------
    centers : array_like

        Numpy array of shape (num_cyl, 3) containing 3-d positions of galaxies.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

        See the :ref:`mock_obs_pos_formatting` documentation page for
        instructions on how to transform your coordinate position arrays into the
        format accepted by the ``galaxies`` and ``particles`` arguments.

    particles : array_like
        Numpy array of shape (num_ptcl, 3) containing 3-d positions of particles.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    particle_masses : array_like
        Float or array of shape (num_ptcl, ) storing the mass of each particle
        in units of Msun with h=1 units. If every particle has the same mass
        (i.e., if your simulation is DM-only), you can pass in a single float.

    downsampling_factor : float
        Factor by which the particles have been randomly downsampled.
        Should be unity if all simulation particles have been chosen.

    rp_bins : array_like
        Numpy array of shape (num_rbins+1, ) of projected radial boundaries
        defining the bins in which the result is calculated.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Length units are assumed to be in Mpc/h, here and throughout Halotools.

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
    total_mass_enclosed : array_like
        Numpy array of shape (num_rbins, ) storing the sum of all particle masses
        enclosed in the input cylinders. Particles appearing in more than one
        cylinder are counted multiple times.

    Examples
    --------
    >>> period = 100.
    >>> num_cyl, num_ptcl = 100, 1000
    >>> centers = np.random.random((num_cyl, 3))*period
    >>> particles = np.random.random((num_ptcl, 3))*period
    >>> masses = np.random.rand(num_ptcl)
    >>> downsampling_factor = 1.
    >>> rp_bins = np.logspace(-1, 1, 15)
    >>> mass_encl = total_mass_enclosed_in_stack_of_cylinders(centers, particles, masses, downsampling_factor, rp_bins, period)
    """

    #  Perform bounds-checking and error-handling in private helper functions
    args = (centers, particles, particle_masses, downsampling_factor,
        rp_bins, period, num_threads)
    result = _enclosed_mass_process_args(*args)
    centers, particles, particle_masses, downsampling_factor, \
        rp_bins, period, num_threads, PBCs = result

    mean_particle_mass = np.mean(particle_masses)
    normalized_particle_masses = particle_masses/mean_particle_mass

    # Calculate M_tot(< Rp) normalized with internal code units
    total_mass_in_stack_of_cylinders = weighted_npairs_xy(centers, particles,
        normalized_particle_masses, rp_bins,
        period=period[:2], num_threads=num_threads,
        approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell2_size)

    # Renormalize the particle masses and account for downsampling
    total_mass_in_stack_of_cylinders *= downsampling_factor*mean_particle_mass

    return total_mass_in_stack_of_cylinders


def total_mass_enclosed_per_cylinder(centers, particles,
        particle_masses, downsampling_factor, rp_bins, period,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r""" Calculate the total mass enclosed in a set of cylinders of infinite length.

    Parameters
    ----------
    centers : array_like

        Numpy array of shape (num_cyl, 3) containing 3-d positions of galaxies.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

        See the :ref:`mock_obs_pos_formatting` documentation page for
        instructions on how to transform your coordinate position arrays into the
        format accepted by the ``galaxies`` and ``particles`` arguments.

    particles : array_like
        Numpy array of shape (num_ptcl, 3) containing 3-d positions of particles.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    particle_masses : array_like
        Float or array of shape (num_ptcl, ) storing the mass of each particle
        in units of Msun with h=1 units. If every particle has the same mass
        (i.e., if your simulation is DM-only), you can pass in a single float.

    downsampling_factor : float
        Factor by which the particles have been randomly downsampled.
        Should be unity if all simulation particles have been chosen.

    rp_bins : array_like
        Numpy array of shape (num_rbins+1, ) of projected radial boundaries
        defining the bins in which the result is calculated.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Length units are assumed to be in Mpc/h, here and throughout Halotools.

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
    total_mass_enclosed : array_like
        Numpy array of shape (num_cyl, num_rbins) storing the sum of all particle masses
        enclosed in each of the input cylinders.

    Examples
    --------
    >>> period = 100.
    >>> num_cyl, num_ptcl = 100, 1000
    >>> centers = np.random.random((num_cyl, 3))*period
    >>> particles = np.random.random((num_ptcl, 3))*period
    >>> masses = np.random.rand(num_ptcl)
    >>> downsampling_factor = 1.
    >>> rp_bins = np.logspace(-1, 1, 15)
    >>> mass_encl = total_mass_enclosed_per_cylinder(centers, particles, masses, downsampling_factor, rp_bins, period)

    The mass enclosed in cylinder i with radius j is given by:

    >>> ith_cylinder_jth_radius_mass = mass_encl[i, j]  # doctest: +SKIP
    """

    #  Perform bounds-checking and error-handling in private helper functions
    args = (centers, particles, particle_masses, downsampling_factor,
        rp_bins, period, num_threads)
    result = _enclosed_mass_process_args(*args)
    centers, particles, particle_masses, downsampling_factor, \
        rp_bins, period, num_threads, PBCs = result

    mean_particle_mass = np.mean(particle_masses)
    normalized_particle_masses = particle_masses/mean_particle_mass

    # Calculate M_tot(< Rp) normalized with internal code units
    total_mass_per_cylinder = weighted_npairs_per_object_xy(centers, particles,
        normalized_particle_masses, rp_bins,
        period=period[:2], num_threads=num_threads,
        approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell2_size)

    # Renormalize the particle masses and account for downsampling
    total_mass_per_cylinder *= downsampling_factor*mean_particle_mass

    return total_mass_per_cylinder


def _enclosed_mass_process_args(centers, particles, masses,
        downsampling_factor, rp_bins, period, num_threads):
    period, PBCs = get_period(period)

    centers = enforce_sample_has_correct_shape(centers)
    particles = enforce_sample_has_correct_shape(particles)

    masses = np.atleast_1d(masses)
    if len(masses) == 1:
        masses = np.zeros(particles.shape[0]) + masses[0]
    else:
        msg = "Must have same number of ``particle_masses`` as particles"
        assert masses.shape[0] == particles.shape[0], msg

    msg = "downsampling_factor = {0} < 1, which is impossible".format(downsampling_factor)
    assert downsampling_factor >= 1, msg

    enforce_sample_respects_pbcs(centers[:, 0], centers[:, 1], centers[:, 2], period)
    enforce_sample_respects_pbcs(particles[:, 0], particles[:, 1], particles[:, 2], period)

    rp_bins = get_separation_bins_array(rp_bins)

    num_threads = get_num_threads(num_threads, enforce_max_cores=False)

    return centers, particles, masses, downsampling_factor, rp_bins, period, num_threads, PBCs
