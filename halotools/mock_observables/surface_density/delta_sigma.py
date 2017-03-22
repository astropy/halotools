"""
Module containing the `~halotools.mock_observables.delta_sigma` function used to
calculate galaxy-galaxy lensing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .surface_density_helpers import annular_area_weighted_midpoints
from .surface_density_helpers import log_interpolation_with_inner_zero_masking as log_interp
from .surface_density_helpers import rho_matter_comoving_in_halotools_units as rho_m_comoving
from .mass_in_cylinders import total_mass_enclosed_in_stack_of_cylinders

from ..mock_observables_helpers import (get_num_threads, get_separation_bins_array,
    get_period, enforce_sample_respects_pbcs, enforce_sample_has_correct_shape)

from ...sim_manager.sim_defaults import default_cosmology

__all__ = ('delta_sigma', 'delta_sigma_from_precomputed_pairs')
__author__ = ('Andrew Hearin', )


def delta_sigma(galaxies, particles, particle_masses, downsampling_factor,
        rp_bins, period, cosmology=default_cosmology, num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate :math:`\Delta\Sigma(r_p)`, the galaxy-galaxy lensing signal
    as a function of projected distance.

    The `delta_sigma` function calculates :math:`\Delta\Sigma(r_p)` by calculating
    the excess surface density of particles in cylinders surrounding the input galaxies.
    The input particles should be a random downsampling of particles in the
    same simulation snapshot as the model galaxies.

    By using the ``particle_masses`` argument, the function works equally well
    with DM-only simulations as with hydro simulations that include
    particles of variable mass.

    Example calls to this function appear in the documentation below.

    See also :ref:`galaxy_catalog_analysis_tutorial3`.

    Parameters
    ----------
    galaxies : array_like

        Numpy array of shape (num_gal, 3) containing 3-d positions of galaxies.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

        See the :ref:`mock_obs_pos_formatting` documentation page for
        instructions on how to transform your coordinate position arrays into the
        format accepted by the ``galaxies`` and ``particles`` arguments.

    particles : array_like
        Numpy array of shape (num_ptcl, 3) containing 3-d positions of particles.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    particle_masses : float or ndarray
        Float or array storing the mass of each particle in units of Msun with h=1 units.

        If passing in an ndarray, must be of shape (num_ptcl, ),
        one array element for every particle.

        If passing in a single float, it will be assumed that every particle
        has the same mass (as is the case in a typical DM-only simulation).

    downsampling_factor : float
        Factor by which the particles have been randomly downsampled.
        Should be unity if all simulation particles have been chosen.

        See the Examples section below for how this can be calculated
        from Halotools-provided catalogs.

    rp_bins : array_like
        Numpy array of shape (num_rbins, ) of projected radial boundaries
        defining the bins in which the result is calculated.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.

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
    rp_mids : array_like
        Numpy array of shape (num_rbins-1, ) storing the projected radii at which
        `Delta_Sigma` has been evaluated.

    Delta_Sigma : array_like
        Numpy array of shape (num_rbins-1, ) storing :math:`\Delta\Sigma(r_p)`
        in comoving units of :math:`h M_{\odot} / {\rm Mpc}^2` assuming h=1.

    Examples
    --------
    For demonstration purposes we will calculate `delta_sigma` using a mock
    catalog generated with the `~halotools.sim_manager.FakeSim`
    that is generated on-the-fly.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    Now let's populate this halo catalog with mock galaxies.

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model = PrebuiltHodModelFactory('leauthaud11', threshold = 11.)
    >>> model.populate_mock(halocat)

    Now we retrieve the positions of our mock galaxies and transform the arrays
    into the shape of the ndarray expected by the `~halotools.mock_observables.delta_sigma`
    function. We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> x = model.mock.galaxy_table['x']
    >>> y = model.mock.galaxy_table['y']
    >>> z = model.mock.galaxy_table['z']
    >>> galaxies = np.vstack((x, y, z)).T

    The `~halotools.mock_observables.return_xyz_formatted_array` function
    also performs this same transformation, and can also be used to place mock
    galaxies into redshift-space for additional observational realism.

    Let's do the same thing for a set of particle data:

    >>> px = model.mock.ptcl_table['x']
    >>> py = model.mock.ptcl_table['y']
    >>> pz = model.mock.ptcl_table['z']
    >>> particles = np.vstack((px, py, pz)).T

    The default Halotools catalogs come with ~1e6 particles.
    Using this many particles may be overkill: in many typical use-cases,
    the `delta_sigma` function converges at the percent-level using
    an order of magnitude fewer particles.
    The code below shows how to (optionally) downsample these particles
    using a Halotools convenience function.

    >>> from halotools.utils import randomly_downsample_data
    >>> num_ptcls_to_use = int(1e4)
    >>> particles = randomly_downsample_data(particles, num_ptcls_to_use)
    >>> particle_masses = np.zeros(num_ptcls_to_use) + halocat.particle_mass

    Whether or not you perform additional downsampling, you will need to account
    for the fact that you are not using the entire snapshot of particles by
    providing the ``downsampling_factor`` argument:

    >>> total_num_ptcl_in_snapshot = halocat.num_ptcl_per_dim**3
    >>> downsampling_factor = total_num_ptcl_in_snapshot/float(len(particles))

    >>> rp_bins = np.logspace(-1, 1, 10)
    >>> period = model.mock.Lbox
    >>> rp_mids, ds = delta_sigma(galaxies, particles, particle_masses, downsampling_factor, rp_bins, period)

    Take care with the units. The values for :math:`\Delta\Sigma` returned by
    the `delta_sigma` functions are in *comoving* units of
    :math:`h M_{\odot} / {\rm Mpc}^2` assuming h=1,
    whereas the typical units used to plot :math:`\Delta\Sigma` are in
    *physical* units of :math:`M_{\odot} / {\rm pc}^2` using the value of
    little h appropriate for your assumed cosmology.

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial3`

    """

    #  Perform bounds-checking and error-handling in private helper functions
    args = (galaxies, particles, particle_masses, downsampling_factor,
        rp_bins, period, num_threads)
    result = _delta_sigma_process_args(*args)
    galaxies, particles, particle_masses, downsampling_factor, \
        rp_bins, period, num_threads, PBCs = result

    total_mass_in_stack_of_cylinders = total_mass_enclosed_in_stack_of_cylinders(
        galaxies, particles, particle_masses, downsampling_factor, rp_bins, period,
        num_threads=num_threads, approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell2_size)
    total_mass_in_stack_of_annuli = np.diff(total_mass_in_stack_of_cylinders)

    mean_rho_comoving = rho_m_comoving(cosmology)
    mean_sigma_comoving = mean_rho_comoving*float(period[2])

    short_funcname = _expected_mass_enclosed_in_random_stack_of_cylinders
    expected_mass_in_random_stack_of_cylinders = short_funcname(
        galaxies.shape[0], period[2], rp_bins, mean_rho_comoving)

    short_funcname = _expected_mass_enclosed_in_random_stack_of_annuli
    expected_mass_in_random_stack_of_annuli = short_funcname(
        galaxies.shape[0], period[2], rp_bins, mean_rho_comoving)

    one_plus_mean_sigma_inside_rp = mean_sigma_comoving*(
        total_mass_in_stack_of_cylinders/expected_mass_in_random_stack_of_cylinders)

    one_plus_sigma = mean_sigma_comoving*(
        total_mass_in_stack_of_annuli/expected_mass_in_random_stack_of_annuli)

    rp_mids = annular_area_weighted_midpoints(rp_bins)
    one_plus_mean_sigma_inside_rp_interp = log_interp(one_plus_mean_sigma_inside_rp,
        rp_bins, rp_mids)

    excess_surface_density = one_plus_mean_sigma_inside_rp_interp - one_plus_sigma
    return rp_mids, excess_surface_density


def delta_sigma_from_precomputed_pairs(galaxies, mass_enclosed_per_galaxy,
        rp_bins, period, cosmology=default_cosmology):
    r"""
    Calculate :math:`\Delta\Sigma(r_p)`, the galaxy-galaxy lensing signal
    as a function of projected distance, assuming the mass around each
    galaxy has been precomputed.

    By using the ``particle_masses`` argument, the function works equally well
    with DM-only simulations as with hydro simulations that include
    particles of variable mass.

    Example calls to this function appear in the documentation below.

    See also :ref:`galaxy_catalog_analysis_tutorial3`.

    Parameters
    ----------
    galaxies : array_like

        Numpy array of shape (num_gal, 3) containing 3-d positions of galaxies.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

        See the :ref:`mock_obs_pos_formatting` documentation page for
        instructions on how to transform your coordinate position arrays into the
        format accepted by the ``galaxies`` and ``particles`` arguments.

    mass_enclosed_per_galaxy : array_like
        Numpy array of shape (num_gal, num_rp_bins+1) storing the mass enclosed inside
        each of the cylinders defined by the input ``rp_bins``.

        The ``mass_enclosed_per_galaxy`` argument can be calculated using the
        `~halotools.mock_observables.total_mass_enclosed_per_cylinder`
        function, as demonstrated in the Examples section below.

    rp_bins : array_like
        Numpy array of shape (num_rbins+1, ) of projected radial boundaries
        defining the bins in which the result is calculated.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Length units are assumed to be in Mpc/h, here and throughout Halotools.

    Returns
    -------
    Delta_Sigma : array_like
        Numpy array of shape (num_rbins-1, ) storing :math:`\Delta\Sigma(r_p)`
        in comoving units of :math:`h M_{\odot} / {\rm Mpc}^2` assuming h=1.

    Examples
    --------
    For demonstration purposes we will calculate `delta_sigma_from_precomputed_pairs`
    using a mock catalog generated with the `~halotools.sim_manager.FakeSim`
    that is generated on-the-fly.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    Now let's populate this halo catalog with mock galaxies.

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model = PrebuiltHodModelFactory('leauthaud11', threshold = 11.)
    >>> model.populate_mock(halocat)

    Now we retrieve the positions of our mock galaxies and transform the arrays
    into the shape of the ndarray expected by the `~halotools.mock_observables.delta_sigma`
    function. We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> x = model.mock.galaxy_table['x']
    >>> y = model.mock.galaxy_table['y']
    >>> z = model.mock.galaxy_table['z']
    >>> galaxies = np.vstack((x, y, z)).T

    The `~halotools.mock_observables.return_xyz_formatted_array` function
    also performs this same transformation, and can also be used to place mock
    galaxies into redshift-space for additional observational realism.

    Let's do the same thing for a set of particle data:

    >>> px = model.mock.ptcl_table['x']
    >>> py = model.mock.ptcl_table['y']
    >>> pz = model.mock.ptcl_table['z']
    >>> particles = np.vstack((px, py, pz)).T

    The default Halotools catalogs come with ~1e6 particles.
    Using this many particles may be overkill: in many typical use-cases,
    the `delta_sigma` function converges at the percent-level using
    an order of magnitude fewer particles.
    The code below shows how to (optionally) downsample these particles
    using a Halotools convenience function.

    >>> from halotools.utils import randomly_downsample_data
    >>> num_ptcls_to_use = int(1e4)
    >>> particles = randomly_downsample_data(particles, num_ptcls_to_use)
    >>> particle_masses = np.zeros(num_ptcls_to_use) + halocat.particle_mass

    Whether or not you perform additional downsampling, you will need to account
    for the fact that you are not using the entire snapshot of particles by
    providing the ``downsampling_factor`` argument:

    >>> total_num_ptcl_in_snapshot = halocat.num_ptcl_per_dim**3
    >>> downsampling_factor = total_num_ptcl_in_snapshot/float(len(particles))

    >>> rp_bins = np.logspace(-1, 1, 10)
    >>> period = model.mock.Lbox

    Now we calculate the mass enclosed by the cylinders around each galaxy:

    >>> from halotools.mock_observables import total_mass_enclosed_per_cylinder

    >>> mass_encl = total_mass_enclosed_per_cylinder(galaxies, particles, particle_masses, downsampling_factor, rp_bins, period)

    At this point, we know the total mass enclosing every galaxy in our sample.
    Now suppose we are only interested in calculating the lensing signal around
    some subsample of our galaxies. Then we just build a mask
    for the sample of galaxies we are interested in and proceed as follows:

    >>> cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
    >>> cens = galaxies[cenmask]
    >>> mass_encl_cens = mass_encl[cenmask, :]

    >>> rp, ds = delta_sigma_from_precomputed_pairs(cens, mass_encl_cens, rp_bins, period, cosmology=halocat.cosmology)

    Take care with the units. The values for :math:`\Delta\Sigma` returned by
    the `delta_sigma` functions are in *comoving* units of
    :math:`h M_{\odot} / {\rm Mpc}^2` assuming h=1,
    whereas the typical units used to plot :math:`\Delta\Sigma` are in
    *physical* units of :math:`M_{\odot} / {\rm pc}^2` using the value of
    little h appropriate for your assumed cosmology.

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial3`

    """

    #  Perform bounds-checking and error-handling in private helper functions
    args = (galaxies, mass_enclosed_per_galaxy, rp_bins, period)
    result = _delta_sigma_precomputed_process_args(*args)
    galaxies, mass_enclosed_per_galaxy, rp_bins, period, PBCs = result

    total_mass_in_stack_of_cylinders = np.sum(mass_enclosed_per_galaxy, axis=0)

    total_mass_in_stack_of_annuli = np.diff(total_mass_in_stack_of_cylinders)

    mean_rho_comoving = rho_m_comoving(cosmology)
    mean_sigma_comoving = mean_rho_comoving*float(period[2])

    short_funcname = _expected_mass_enclosed_in_random_stack_of_cylinders
    expected_mass_in_random_stack_of_cylinders = short_funcname(
        galaxies.shape[0], period[2], rp_bins, mean_rho_comoving)

    short_funcname = _expected_mass_enclosed_in_random_stack_of_annuli
    expected_mass_in_random_stack_of_annuli = short_funcname(
        galaxies.shape[0], period[2], rp_bins, mean_rho_comoving)

    one_plus_mean_sigma_inside_rp = mean_sigma_comoving*(
        total_mass_in_stack_of_cylinders/expected_mass_in_random_stack_of_cylinders)

    one_plus_sigma = mean_sigma_comoving*(
        total_mass_in_stack_of_annuli/expected_mass_in_random_stack_of_annuli)

    rp_mids = annular_area_weighted_midpoints(rp_bins)
    one_plus_mean_sigma_inside_rp_interp = log_interp(one_plus_mean_sigma_inside_rp,
        rp_bins, rp_mids)

    excess_surface_density = one_plus_mean_sigma_inside_rp_interp - one_plus_sigma
    return rp_mids, excess_surface_density


def _delta_sigma_precomputed_process_args(galaxies, mass_enclosed_per_galaxy, rp_bins, period):
    period, PBCs = get_period(period)
    galaxies = enforce_sample_has_correct_shape(galaxies)
    enforce_sample_respects_pbcs(galaxies[:, 0], galaxies[:, 1], galaxies[:, 2], period)
    rp_bins = get_separation_bins_array(rp_bins)

    num_gals, num_rp_bins = galaxies.shape[0], rp_bins.shape[0]
    msg = "Shape of input ``mass_enclosed_per_galaxy`` must be ({0}, {1})".format(
        num_gals, num_rp_bins)
    assert mass_enclosed_per_galaxy.shape == (num_gals, num_rp_bins), msg

    return galaxies, mass_enclosed_per_galaxy, rp_bins, period, PBCs


def _delta_sigma_process_args(galaxies, particles, masses, downsampling_factor,
        rp_bins, period, num_threads):
    period, PBCs = get_period(period)

    galaxies = enforce_sample_has_correct_shape(galaxies)
    particles = enforce_sample_has_correct_shape(particles)

    masses = np.atleast_1d(masses)
    if len(masses) == 1:
        masses = np.zeros(particles.shape[0]) + masses[0]
    else:
        msg = "Must have same number of ``particle_masses`` as particles"
        assert masses.shape[0] == particles.shape[0], msg

    msg = "downsampling_factor = {0} < 1, which is impossible".format(downsampling_factor)
    assert downsampling_factor >= 1, msg

    enforce_sample_respects_pbcs(galaxies[:, 0], galaxies[:, 1], galaxies[:, 2], period)
    enforce_sample_respects_pbcs(particles[:, 0], particles[:, 1], particles[:, 2], period)

    rp_bins = get_separation_bins_array(rp_bins)

    num_threads = get_num_threads(num_threads, enforce_max_cores=False)

    return galaxies, particles, masses, downsampling_factor, rp_bins, period, num_threads, PBCs


def _expected_mass_enclosed_in_random_stack_of_cylinders(num_total_cylinders,
        Lbox, rp_bins, mean_rho_comoving):

    cylinder_volumes = Lbox*np.pi*rp_bins**2
    expected_mass_in_random_cylinder = mean_rho_comoving*cylinder_volumes
    return expected_mass_in_random_cylinder*num_total_cylinders


def _expected_mass_enclosed_in_random_stack_of_annuli(num_total_annuli,
        Lbox, rp_bins, mean_rho_comoving):

    annuli_volumes = Lbox*np.pi*np.diff(rp_bins**2)
    expected_mass_in_random_annulus = mean_rho_comoving*annuli_volumes
    return expected_mass_in_random_annulus*num_total_annuli
