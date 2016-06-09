"""
Module containing the `~halotools.mock_observables.delta_sigma` function used to
calculate galaxy-galaxy lensing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
from warnings import warn

from astropy import units as u
from astropy.constants import G

from .tpcf import tpcf
from .clustering_helpers import verify_tpcf_estimator

from ..mock_observables_helpers import (get_num_threads, get_separation_bins_array,
    get_period, enforce_sample_respects_pbcs, enforce_sample_has_correct_shape)

from ...sim_manager.sim_defaults import default_cosmology

__all__ = ['delta_sigma']
__author__ = ['Duncan Campbell']

newtonG = G.to(u.km*u.km*u.Mpc/(u.Msun*u.s*u.s))


def delta_sigma(galaxies, particles, rp_bins, pi_max, period,
        cosmology=default_cosmology,
        log_bins=True, n_bins=25, estimator='Natural', num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None):
    """
    Calculate the galaxy-galaxy lensing signal :math:`\\Delta\\Sigma(r_p)` as a function
    of projected distance.

    This function first computes the cross correlation between ``galaxies`` and ``particles``
    to get the galaxy-matter cross correlation, :math:`\\xi_{\\rm g, m}(r)`.
    Then the function performs a projection integral of :math:`\\xi_{\\rm g, m}(r)`
    to get :math:`\\Delta\\Sigma(r_p)`.  See the notes for details
    about the calculation.

    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``galaxies`` and ``particles`` arguments.

    See also :ref:`galaxy_catalog_analysis_tutorial3`.

    Parameters
    ----------
    galaxies : array_like
        Ngal x 3 numpy array containing 3-d positions of galaxies.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    particles : array_like
        Npart x 3 numpy array containing 3-d positions of particles.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    rp_bins : array_like
        array of projected radial boundaries defining the bins in which the result is
        calculated.  The minimum of rp_bins must be > 0.0.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    pi_max: float
        maximum integration parameter, :math:`\\pi_{\\rm max}`
        (see notes for more details).
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    cosmology : instance of `astropy.cosmology`, optional
        Default value is set in `~halotools.sim_manager.default_cosmology` module.
        Typically you should use the `cosmology` attribute of the halo catalog
        you used to populate mock galaxies.

    period : array_like
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    log_bins : boolean, optional
        integration parameter (see notes for more details).

    n_bins : int, optional
        integration parameter (see notes for more details).

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

    Returns
    -------
    Delta_Sigma : np.array
        :math:`\\Delta\\Sigma(r_p)` calculated at projected comoving radial distances ``rp_bins``.
        The units of `ds` are :math:`h * M_{\odot} / Mpc^2`, where distances are in comoving units.
        You can convert to physical units using the input cosmology and redshift.
        Note that little h = 1 here and throughout Halotools.

    Notes
    -----
    :math:`\\Delta\\Sigma` is calculated by first calculating,

    .. math::
        \\Sigma(r_p) = 2.0 * \\bar{\\rho}\\int_0^{\\pi_{\\rm max}} \\left[1+\\xi_{\\rm g,m}(\\sqrt{r_p^2+\\pi^2}) \\right]\\mathrm{d}\\pi

    and then,

    .. math::
        \\Delta\\Sigma(r_p) = \\bar{\\Sigma}(<r_p) - \\Sigma(r_p)

    where,

    .. math::
        \\bar{\\Sigma}(<r_p) = \\frac{1}{\\pi r_p^2}\\int_0^{r_p}\\Sigma(r_p^{\\prime})2\\pi r_p^{\\prime} \\mathrm{d}r_p^{\\prime}

    Numerically, :math:`\\xi_{\\rm g,m}` is calculated in `n_bins` evenly spaced linearly
    or log-linearly as indicated by `log_bins` and integrated between
    :math:`{\\rm rp}_{\\rm min}` and
    :math:`\\sqrt{{\\rm{rp}_{\\rm max}}^2 + {\\pi_{\\rm max}}^2}`.

    All integrals are done use `scipy.integrate.quad`.

    Users of the `~halotools.mock_observables.delta_sigma` function should be aware that
    the current halotools implementation is only one method of calculation for gg-lensing.
    One alternative would be to do the entire calculation fully in 2d, using the entire
    simulation z-axis as the line of sight (this is the approach taken in
    Hearin et al. 2013, http://arxiv.org/abs/1310.6747). Determing the optimal
    calculation method for :math:`\Delta\Sigma` is a subject of ongoing research,
    so if you want to use this function for your own science application, be sure
    to read the source code implementation and test that the implementation is
    sufficient for your needs.

    Examples
    --------
    For demonstration purposes we will calculate `delta_sigma` using the
    `~halotools.sim_manager.FakeSim` that is generated on-the-fly.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    Now let's populate this halo catalog with mock galaxies.

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model = PrebuiltHodModelFactory('hearin15', threshold = 11.5)
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

    Let's do the same thing for a set of particle data

    >>> px = model.mock.ptcl_table['x']
    >>> py = model.mock.ptcl_table['y']
    >>> pz = model.mock.ptcl_table['z']
    >>> particles = np.vstack((px, py, pz)).T

    The default Halotools catalogs come with about one million particles.
    The code below shows how to (optionally) downsample using a Halotools
    convenience function. This is just for demonstration purposes. For a real
    analysis, you should use at least as many dark matter particles as galaxies.

    >>> from halotools.utils import randomly_downsample_data
    >>> particles = randomly_downsample_data(particles, int(5e3))

    >>> rp_bins = np.logspace(-1, 1, 10)
    >>> pi_max = 15
    >>> period = model.mock.Lbox
    >>> cosmology = halocat.cosmology
    >>> ds = delta_sigma(galaxies, particles, rp_bins, pi_max, period, cosmology=cosmology)

    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial3`

    """

    #process the input parameters
    args = (galaxies, particles, rp_bins, period, estimator, num_threads)
    result = _delta_sigma_process_args(*args)
    galaxies, particles, rp_bins, period, estimator, num_threads, PBCs = result

    #determine radial bins to calculate tpcf in
    rp_max = np.max(rp_bins)
    rp_min = np.min(rp_bins)
    #maximum radial distance to calculate TPCF out to:
    rmax = np.sqrt(rp_max**2 + pi_max**2)

    #define radial bins using either log or linear spacing
    if log_bins is True:
        rbins = np.logspace(np.log10(rp_min), np.log10(rmax), n_bins)
    else:
        rbins = np.linspace(rp_min, rmax, n_bins)

    #calculate the cross-correlation between galaxies and particles
    xi = tpcf(galaxies, rbins, sample2=particles, randoms=None, period=period,
        do_auto=False, do_cross=True, estimator=estimator, num_threads=num_threads,
        approx_cell1_size=approx_cell1_size, approx_cell2_size=approx_cell2_size)

    #Check to see if xi ever is equal to -1
    #if so, there are radial bins with 0 matter particles.
    #This could mean that the user has under-sampled the particles.
    if np.any(xi==-1.0):
        msg = ("\n"
               "Some radial bins contain 0 particles in the \n"
               "galaxy-matter cross cross correlation calculation. \n"
               "If you downsampled the amount of matter particles, \n"
               "consider using more particles. Alternatively, you \n"
               "may (also) want to use fewer bins in the calculation. \n"
               "Set the `n_bins` parameter to a smaller number. \n"
               "Finally, you may want to calculate Delta_Sigma to \n"
               "a larger minimum projected distance, min(`rp_bins`)."
               )
        warn(msg)

    # fit a spline to the tpcf
    # note that we fit log10(1 + xi)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0  # note these are the true centers, not log
    xi = InterpolatedUnivariateSpline(rbin_centers, np.log10(xi+1.0), ext=0)

    rho_crit0 = cosmology.critical_density0
    rho_crit0 = rho_crit0.to(u.Msun/u.Mpc**3).value/cosmology.h**2
    mean_rho_comoving = cosmology.Om0*rho_crit0

    # define function to integrate
    def twice_one_plus_xi_gm(pi, rp):
        r = np.sqrt(rp**2+pi**2)
        # note that we take 10**xi-1,
        # because we fit the log10(1 + xi)
        return 2.0*(1.0+(10.0**xi(r)-1.0))

    # integrate xi to get the surface density as a function of r_p
    dimless_surface_density = list(
        integrate.quad(twice_one_plus_xi_gm, 0.0, pi_max, args=(rp,))[0] for rp in rp_bins)

    # fit a spline to the surface density
    log10_dimless_surface_density = InterpolatedUnivariateSpline(
        rp_bins, np.log10(dimless_surface_density), ext=0)

    # integrate surface density to get the mean internal surface density
    # define function to integrate
    def dimless_mean_internal_surface_density_integrand(rp):
        # note that we take 10**surface_density,
        # because we fit the log of surface density
        return 10.0**log10_dimless_surface_density(rp)*2.0*np.pi*rp

    # do integral to get mean internal surface density
    dimless_mean_internal_surface_density = np.zeros(len(rp_bins))
    for i in range(0, len(rp_bins)):
        internal_area = np.pi*rp_bins[i]**2.0
        dimless_mean_internal_surface_density[i] = integrate.quad(
            dimless_mean_internal_surface_density_integrand, 0.0, rp_bins[i])[0]/(internal_area)

    # calculate an return the change in surface density, delta sigma
    dimless_delta_sigma = dimless_mean_internal_surface_density - 10**log10_dimless_surface_density(rp_bins)

    return dimless_delta_sigma*mean_rho_comoving


def _delta_sigma_process_args(galaxies, particles, rp_bins, period, estimator, num_threads):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.delta_sigma`.
    """
    period, PBCs = get_period(period)
    if PBCs is False:
        msg = ("The `delta_sigma` function requires the input ``period`` to be \n"
            "a bounded positive number in all dimensions")
        raise ValueError(msg)

    galaxies = enforce_sample_has_correct_shape(galaxies)
    particles = enforce_sample_has_correct_shape(particles)

    enforce_sample_respects_pbcs(galaxies[:, 0], galaxies[:, 1], galaxies[:, 2], period)
    enforce_sample_respects_pbcs(particles[:, 0], particles[:, 1], particles[:, 2], period)

    rp_bins = get_separation_bins_array(rp_bins)

    num_threads = get_num_threads(num_threads, enforce_max_cores=False)

    estimator = verify_tpcf_estimator(estimator)

    return galaxies, particles, rp_bins, period, estimator, num_threads, PBCs
