"""
Module containing the `~halotools.mock_observables.mean_delta_sigma` function
used to calculate galaxy-galaxy lensing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from functools import partial
import multiprocessing

from .engines import mean_delta_sigma_engine

from ..mock_observables_helpers import (get_num_threads, get_separation_bins_array,
    get_period, enforce_sample_respects_pbcs, enforce_sample_has_correct_shape)
from ..pair_counters.rectangular_mesh_2d import RectangularDoubleMesh2D
from ..pair_counters.mesh_helpers import _set_approximate_2d_cell_sizes
from ..pair_counters.mesh_helpers import _cell1_parallelization_indices
from ..pair_counters.mesh_helpers import _enclose_in_square

from ...utils.array_utils import custom_len

__all__ = ('mean_delta_sigma', )
__author__ = ('Andrew Hearin', 'Johannes Ulf Lange')


def mean_delta_sigma(galaxies, particles, effective_particle_masses,
                     rp_bins, period=None, verbose=False, num_threads=1,
                     approx_cell1_size=None, approx_cell2_size=None,
                     per_object=False):
    r"""
    Calculate :math:`\Delta\Sigma(r_p)`, the galaxy-galaxy lensing signal
    as a function of projected distance.

    The `mean_delta_sigma` function calculates :math:`\Delta\Sigma(r_p)` by calculating
    the excess surface density of particles in cylinders surrounding the input galaxies.
    The input particles should be a random downsampling of particles in the
    same simulation snapshot as the model galaxies.

    By using the ``effective_particle_masses`` argument, the function works equally well
    with DM-only simulations as with hydro simulations that include
    multiple species of particles with different masses and/or different downsampling rates.

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

    effective_particle_masses : float or ndarray
        Float or array storing the effective mass of each particle in units of Msun with h=1 units.

        If passing in an ndarray, must be of shape (num_ptcl, ),
        one array element for every particle.

        If passing in a single float, it will be assumed that every particle
        has the same mass (as is the case in a typical DM-only simulation).

        The effective mass is simply the actual mass multiplied by the downsampling rate.
        For example, if your simulation has a particle mass of 10**8 and you are using a
        sample of particles that have been randomly downsampled at a 1% rate, then
        your effective particle mass will be 10**10.

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

    per_object : bool, optional
        Boolean flag specifying whether the function will return the per-object
        lensing signal. Default is False, in which the returned array will be
        an average over the entire sample. If True, the returned ndarray will
        have shape (num_gal, num_rbins)

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
    Delta_Sigma : array_like
        Numpy array of shape (num_rbins-1, ) storing :math:`\Delta\Sigma(r_p)`
        in comoving units of :math:`h M_{\odot} / {\rm Mpc}^2` assuming h=1.

        If per_object is True, Delta_Sigma will instead have shape (num_gal, num_rbins)

    Examples
    --------
    For demonstration purposes we will calculate `mean_delta_sigma` using a mock
    catalog generated with the `~halotools.sim_manager.FakeSim`
    that is generated on-the-fly.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    Now let's populate this halo catalog with mock galaxies.

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model = PrebuiltHodModelFactory('leauthaud11', threshold = 11.)
    >>> model.populate_mock(halocat)

    Now we retrieve the positions of our mock galaxies and transform the arrays
    into the shape of the ndarray expected by the `~halotools.mock_observables.mean_delta_sigma`
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
    the `mean_delta_sigma` function converges at the percent-level using
    an order of magnitude fewer particles.
    The code below shows how to (optionally) downsample these particles
    using a Halotools convenience function.

    >>> from halotools.utils import randomly_downsample_data
    >>> num_ptcls_to_use = int(1e4)
    >>> particles = randomly_downsample_data(particles, num_ptcls_to_use)
    >>> particle_masses = np.zeros(num_ptcls_to_use) + halocat.particle_mass

    >>> total_num_ptcl_in_snapshot = halocat.num_ptcl_per_dim**3
    >>> downsampling_factor = total_num_ptcl_in_snapshot/float(len(particles))
    >>> effective_particle_masses = downsampling_factor * particle_masses

    >>> rp_bins = np.logspace(-1, 1, 10)
    >>> period = model.mock.Lbox
    >>> ds = mean_delta_sigma(galaxies, particles, effective_particle_masses, rp_bins, period)

    Take care with the units. The values for :math:`\Delta\Sigma` returned by
    the `mean_delta_sigma` functions are in *comoving* units of
    :math:`h M_{\odot} / {\rm Mpc}^2` assuming h=1,
    whereas the typical units used to plot :math:`\Delta\Sigma` are in
    *physical* units of :math:`M_{\odot} / {\rm pc}^2` using the value of
    little h appropriate for your assumed cosmology.

    The code shown above demonstrates how to calculate :math:`\Delta\Sigma` via the excess
    surface density of mass using the z-axis as the axis of projection. However, it may be useful
    to project along the other Cartesian axes, for example to help beat down sample variance.
    While the `mean_delta_sigma` function is written to always use the "third" dimension as the
    projection axis, you can easily hack the code to project along, say, the y-axis by simply
    transposing your y- and z-coordinates when you pack them into a 2-d array:

    >>> particles = np.vstack((px, pz, py)).T
    >>> galaxies = np.vstack((x, z, y)).T

    Using the above ``particles`` and ``galaxies`` and otherwise calling the `mean_delta_sigma`
    function as normal will instead calculate the surface mass density by projecting
    along the y-axis.

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial3`

    """
    # Process the inputs with the helper function
    result = _mean_delta_sigma_process_args(
        galaxies, particles, effective_particle_masses, rp_bins,
        period, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, x2in, y2in, w2in = result[0:5]
    rp_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[5:]
    xperiod, yperiod = period[:2]

    rp_max = np.max(rp_bins)
    search_xlength, search_ylength = rp_max, rp_max

    # Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (_set_approximate_2d_cell_sizes(
        approx_cell1_size, approx_cell2_size, period))
    approx_x1cell_size, approx_y1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size = approx_cell2_size

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh2D(
        x1in, y1in, x2in, y2in,
        approx_x1cell_size, approx_y1cell_size,
        approx_x2cell_size, approx_y2cell_size,
        search_xlength, search_ylength, xperiod, yperiod, PBCs)

    # Create a function object that has a single argument, for parallelization
    # purposes
    counting_engine = partial(mean_delta_sigma_engine, double_mesh, x1in, y1in,
                              x2in, y2in, w2in, rp_bins)

    # # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(counting_engine, cell1_tuples)
        delta_sigma = np.sum(np.array(result), axis=0)
        pool.close()
    else:
        delta_sigma = counting_engine(cell1_tuples[0])

    if per_object:
        return delta_sigma
    else:
        return np.mean(delta_sigma, axis=0)


def _mean_delta_sigma_process_args(
        galaxies, particles, effective_particle_masses, rp_bins,
        period, num_threads, approx_cell1_size, approx_cell2_size):

    period, PBCs = get_period(period)
    if PBCs is False:
        _result = _enclose_in_box(
                galaxies[:, 0], galaxies[:, 1], galaxies[:, 2],
                particles[:, 0], particles[:, 1], particles[:, 2])
        _x1, _y1, _z1, _x2, _y2, _z2, period = _result
        galaxies[:, 0] = _x1
        galaxies[:, 1] = _y1
        galaxies[:, 2] = _z1
        particles[:, 0] = _x2
        particles[:, 1] = _y2
        particles[:, 2] = _z2

    galaxies = enforce_sample_has_correct_shape(galaxies)
    particles = enforce_sample_has_correct_shape(particles)

    effective_particle_masses = np.atleast_1d(effective_particle_masses)
    if len(effective_particle_masses) == 1:
        effective_particle_masses = np.zeros(particles.shape[0]) + effective_particle_masses[0]
    else:
        msg = "Must have same number of ``particle_masses`` as particles"
        assert effective_particle_masses.shape[0] == particles.shape[0], msg

    enforce_sample_respects_pbcs(galaxies[:, 0], galaxies[:, 1],
                                 galaxies[:, 2], period)
    enforce_sample_respects_pbcs(particles[:, 0], particles[:, 1],
                                 particles[:, 2], period)

    x1 = galaxies[:, 0]
    y1 = galaxies[:, 1]
    x2 = particles[:, 0]
    y2 = particles[:, 1]

    rp_bins = get_separation_bins_array(rp_bins)
    rp_max = np.max(rp_bins)

    if period is None:
        PBCs = False
        x1, y1, x2, y2, period = (
            _enclose_in_square(x1, y1, x2, y2,
                               min_size=[rp_max*3.0, rp_max*3.0]))

    num_threads = get_num_threads(num_threads, enforce_max_cores=False)

    if approx_cell1_size is None:
        approx_cell1_size = [rp_max, rp_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [rp_max, rp_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size]

    return (x1, y1, x2, y2, effective_particle_masses, rp_bins,
            period, num_threads, PBCs, approx_cell1_size, approx_cell2_size)


def _enclose_in_box(x1, y1, z1, x2, y2, z2, min_size=None):
    """
    Build box which encloses all points, shifting the points so that
    the "leftmost" point is (0,0,0).

    Parameters
    ----------
    x1,y1,z1 : array_like
        cartesian positions of points

    x2,y2,z2 : array_like
        cartesian positions of points

    min_size : array_like
        minimum lengths of a side of the box.  If the minimum box constructed around the
        points has a side i less than ``min_size[i]``, then the box is padded in order to
        obtain the minimum specified size.

    Returns
    -------
    x1, y1, z1, x2, y2, z2, Lbox
        shifted positions and box size.
    """
    xmin = np.min([np.min(x1), np.min(x2)])
    ymin = np.min([np.min(y1), np.min(y2)])
    zmin = np.min([np.min(z1), np.min(z2)])
    xmax = np.max([np.max(x1), np.max(x2)])
    ymax = np.max([np.max(y1), np.max(y2)])
    zmax = np.max([np.max(z1), np.max(z2)])

    xyzmin = np.min([xmin, ymin, zmin])
    xyzmax = np.max([xmax, ymax, zmax])-xyzmin

    x1 = x1 - xyzmin
    y1 = y1 - xyzmin
    z1 = z1 - xyzmin
    x2 = x2 - xyzmin
    y2 = y2 - xyzmin
    z2 = z2 - xyzmin

    Lbox = np.array([xyzmax, xyzmax, xyzmax])

    if min_size is not None:
        min_size = np.atleast_1d(min_size)
        if np.any(Lbox < min_size):
            Lbox[(Lbox < min_size)] = min_size[(Lbox < min_size)]

    return x1, y1, z1, x2, y2, z2, Lbox
