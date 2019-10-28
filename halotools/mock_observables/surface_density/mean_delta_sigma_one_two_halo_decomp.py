"""
Module containing the `~halotools.mock_observables.mean_delta_sigma` function
used to calculate galaxy-galaxy lensing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from functools import partial
import multiprocessing

from .engines import mean_ds_12h_halo_id_engine, mean_ds_12h_rhalo_engine

from ..mock_observables_helpers import (get_num_threads, get_separation_bins_array,
    get_period, enforce_sample_respects_pbcs, enforce_sample_has_correct_shape)
from ..pair_counters.rectangular_mesh_2d import RectangularDoubleMesh2D
from ..pair_counters.mesh_helpers import _set_approximate_2d_cell_sizes
from ..pair_counters.mesh_helpers import _cell1_parallelization_indices
from ..pair_counters.mesh_helpers import _enclose_in_square

from ...utils.array_utils import array_is_monotonic, custom_len

__all__ = ('mean_delta_sigma_one_two_halo_decomp', )
__author__ = ('Andrew Hearin', 'Johannes Ulf Lange')


def mean_delta_sigma_one_two_halo_decomp(galaxies, particles, particle_masses,
            downsampling_factor, rp_bins, period=None,
            halo_radii=None, galaxy_halo_ids=None, particle_halo_ids=None,
            verbose=False, num_threads=1,
            approx_cell1_size=None, approx_cell2_size=None,
            per_object=False):
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
    >>> galaxy_halo_ids = np.arange(len(galaxies)).astype(int)
    >>> particle_halo_ids = np.random.randint(0, len(galaxies), len(particles))
    >>> ds_1h, ds_2h = mean_delta_sigma_one_two_halo_decomp(galaxies, particles, particle_masses, downsampling_factor, rp_bins, period, galaxy_halo_ids=galaxy_halo_ids, particle_halo_ids=particle_halo_ids)

    >>> halo_radii = np.random.uniform(0, 0.1, len(galaxies))
    >>> ds_1h, ds_2h = mean_delta_sigma_one_two_halo_decomp(galaxies, particles, particle_masses, downsampling_factor, rp_bins, period, halo_radii=halo_radii)

    Take care with the units. The values for :math:`\Delta\Sigma` returned by
    the `delta_sigma` functions are in *comoving* units of
    :math:`h M_{\odot} / {\rm Mpc}^2` assuming h=1,
    whereas the typical units used to plot :math:`\Delta\Sigma` are in
    *physical* units of :math:`M_{\odot} / {\rm pc}^2` using the value of
    little h appropriate for your assumed cosmology.

    The code shown above demonstrates how to calculate :math:`\Delta\Sigma` via the excess
    surface density of mass using the z-axis as the axis of projection. However, it may be useful
    to project along the other Cartesian axes, for example to help beat down sample variance.
    While the `delta_sigma` function is written to always use the "third" dimension as the
    projection axis, you can easily hack the code to project along, say, the y-axis by simply
    transposing your y- and z-coordinates when you pack them into a 2-d array:

    >>> particles = np.vstack((px, pz, py)).T
    >>> galaxies = np.vstack((x, z, y)).T

    Using the above ``particles`` and ``galaxies`` and otherwise calling the `delta_sigma`
    function as normal will instead calculate the surface mass density by projecting
    along the y-axis.

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial3`

    """
    # Process the inputs with the helper function
    result = _mean_delta_sigma_process_args(
        galaxies, particles, particle_masses, halo_radii, galaxy_halo_ids, particle_halo_ids,
        downsampling_factor, rp_bins,
        period, num_threads, approx_cell1_size, approx_cell2_size)

    x1in, y1in, z1in, x2in, y2in, z2in, w2in = result[0:7]
    halo_radii, galaxy_halo_ids, particle_halo_ids, rp_bins, period = result[7:12]
    num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[12:]

    xperiod, yperiod, zperiod = period

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

    if halo_radii is None:
        counting_engine = partial(mean_ds_12h_halo_id_engine, double_mesh, x1in, y1in,
                    galaxy_halo_ids, x2in, y2in, w2in, particle_halo_ids, rp_bins)
    else:
        counting_engine = partial(mean_ds_12h_rhalo_engine, double_mesh, zperiod, x1in, y1in, z1in, halo_radii,
                    x2in, y2in, z2in, w2in, rp_bins)

    # # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = np.array(pool.map(counting_engine, cell1_tuples))
        ds_1h_per_obj = np.sum(result[:, 0, :, :], axis=0)
        ds_2h_per_obj = np.sum(result[:, 1, :, :], axis=0)
        pool.close()
    else:
        ds_1h_per_obj, ds_2h_per_obj = counting_engine(cell1_tuples[0])

    if per_object:
        return ds_1h_per_obj, ds_2h_per_obj
    else:
        return np.mean(ds_1h_per_obj, axis=0), np.mean(ds_2h_per_obj, axis=0)


def _mean_delta_sigma_process_args(
        galaxies, particles, particle_masses, halo_radii, galaxy_halo_ids, particle_halo_ids,
        downsampling_factor, rp_bins,
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

    particle_masses = np.atleast_1d(particle_masses)
    if len(particle_masses) == 1:
        particle_masses = np.zeros(particles.shape[0]) + particle_masses[0]
    else:
        msg = "Must have same number of ``particle_masses`` as particles"
        assert particle_masses.shape[0] == particles.shape[0], msg

    msg = "downsampling_factor = {0} < 1, which is impossible".format(
        downsampling_factor)
    assert downsampling_factor >= 1, msg

    enforce_sample_respects_pbcs(galaxies[:, 0], galaxies[:, 1],
                                 galaxies[:, 2], period)
    enforce_sample_respects_pbcs(particles[:, 0], particles[:, 1],
                                 particles[:, 2], period)

    x1 = galaxies[:, 0]
    y1 = galaxies[:, 1]
    z1 = galaxies[:, 2]
    x2 = particles[:, 0]
    y2 = particles[:, 1]
    z2 = particles[:, 2]

    if halo_radii is None:
        try:
            galaxy_halo_ids = np.atleast_1d(galaxy_halo_ids).astype('i8')
            particle_halo_ids = np.atleast_1d(particle_halo_ids).astype('i8')
            assert galaxy_halo_ids.shape[0] == x1.shape[0]
        except (TypeError, ValueError):
            msg = "If halo_radii argument is None, must pass both galaxy_halo_ids and particle_halo_ids"
            raise ValueError(msg)
        except AssertionError:
            msg = "Input galaxy_halo_ids must have length as input galaxies"
            raise ValueError(msg)
    elif (galaxy_halo_ids is None) and (particle_halo_ids is None):
        try:
            halo_radii = np.atleast_1d(halo_radii).astype('f8')
            assert halo_radii.shape[0] == x1.shape[0]
        except (TypeError, ValueError):
            msg = "If galaxy_halo_ids is None, must pass halo_radii"
            raise ValueError(msg)
        except AssertionError:
            msg = "Input halo_radii must have length as input galaxies"
            raise ValueError(msg)
    else:
        msg = "Must either pass halo_radii, or alternatively must pass both galaxy_halo_ids and particle_halo_ids"
        raise ValueError(msg)

    rp_bins = get_separation_bins_array(rp_bins)
    rp_max = np.max(rp_bins)

    # if period is None:
    #     PBCs = False
    #     x1, y1, x2, y2, period = (
    #         _enclose_in_square(x1, y1, x2, y2,
    #                            min_size=[rp_max*3.0, rp_max*3.0]))

    num_threads = get_num_threads(num_threads, enforce_max_cores=False)

    if approx_cell1_size is None:
        approx_cell1_size = [rp_max, rp_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [rp_max, rp_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size]

    return (x1, y1, z1, x2, y2, z2, particle_masses * downsampling_factor,
            halo_radii, galaxy_halo_ids, particle_halo_ids, rp_bins,
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
