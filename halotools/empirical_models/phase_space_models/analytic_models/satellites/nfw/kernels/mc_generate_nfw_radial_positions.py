"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .mass_profile import cumulative_mass_PDF

from ....halo_boundary_functions import halo_mass_to_halo_radius

from ......model_helpers import custom_spline
from ......model_defaults import halo_mass_definition as default_halo_mass_definition

from .......sim_manager.sim_defaults import default_cosmology, default_redshift
from .......custom_exceptions import HalotoolsError

__all__ = ('mc_generate_nfw_radial_positions', )


def mc_generate_nfw_radial_positions(num_pts=int(1e4), conc=5,
            cosmology=default_cosmology, redshift=default_redshift,
            mdef=default_halo_mass_definition, seed=None,
            **kwargs):
    r""" Return a Monte Carlo realization of points in an NFW profile.

    See :ref:`monte_carlo_nfw_spatial_profile` for a discussion of this technique.

    Parameters
    -----------
    num_pts : int, optional
        Number of points in the Monte Carlo realization of the profile.
        Default is 1e4.

    conc : float, optional
        Concentration of the NFW profile being realized.
        Default is 5.

    halo_mass : float, optional
        Total mass of the halo whose profile is being realized.

        If ``halo_mass`` is unspecified,
        keyword argument ``halo_radius`` must be specified.

    halo_radius : float, optional
        Physical boundary of the halo whose profile is being realized
        in units of Mpc/h.

        If ``halo_radius`` is unspecified,
        keyword argument ``halo_mass`` must be specified, in which case the
        outer boundary of the halo will be determined
        according to the selected mass definition

    cosmology : object, optional
        Instance of an Astropy `~astropy.cosmology` object.
        Default is set in `~halotools.sim_manager.sim_defaults`

    redshift: array_like, optional
        Can either be a scalar, or a numpy array of the same dimension as the input ``halo_mass``.
        Default is set in `~halotools.sim_manager.sim_defaults`

    mdef: str
        String specifying the halo mass definition, e.g., 'vir' or '200m'.
        Default is set in `~halotools.empirical_models.model_defaults`

    seed : int, optional
        Random number seed used in the Monte Carlo realization.
        Default is None, which will produce stochastic results.

    Returns
    --------
    radial_positions : array_like
        Numpy array storing a Monte Carlo realization of the halo profile.
        All values will lie strictly between 0 and the halo boundary.

    Examples
    ---------
    >>> radial_positions = mc_generate_nfw_radial_positions(halo_mass = 1e12, conc = 10)
    >>> radial_positions = mc_generate_nfw_radial_positions(halo_radius = 0.25)
    """
    try:
        halo_radius = kwargs['halo_radius']
    except KeyError:
        try:
            halo_mass = kwargs['halo_mass']
            halo_radius = halo_mass_to_halo_radius(halo_mass, cosmology, redshift, mdef)
        except KeyError:
            msg = ("\nIf keyword argument ``halo_radius`` is unspecified, "
                "argument ``halo_mass`` must be specified.\n")
            raise HalotoolsError(msg)
        except TypeError:
            raise HalotoolsError("Input ``halo_mass`` must be a float")

    halo_radius = np.atleast_1d(halo_radius).astype(np.float64)
    try:
        assert len(halo_radius) == 1
    except AssertionError:
        msg = ("Input ``halo_radius`` must be a float")
        raise HalotoolsError(msg)

    conc = np.atleast_1d(conc).astype(np.float64)
    try:
        assert len(conc) == 1
    except AssertionError:
        msg = ("Input ``conc`` must be a float")
        raise HalotoolsError(msg)

    # Build lookup table from which to tabulate the inverse cumulative_mass_PDF
    Npts_radius_table = int(1e3)
    radius_array = np.logspace(-4, 0, Npts_radius_table)
    logradius_array = np.log10(radius_array)
    table_ordinates = cumulative_mass_PDF(radius_array, conc)
    log_table_ordinates = np.log10(table_ordinates)
    funcobj = custom_spline(log_table_ordinates, logradius_array, k=3)

    # Use method of Inverse Transform Sampling to generate a Monte Carlo realization
    # of the radial positions
    with NumpyRNGContext(seed):
        randoms = np.random.uniform(0, 1, num_pts)
    log_randoms = np.log10(randoms)
    log_scaled_radial_positions = funcobj(log_randoms)
    scaled_radial_positions = 10.**log_scaled_radial_positions
    radial_positions = scaled_radial_positions*halo_radius

    return radial_positions
