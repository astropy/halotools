"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .pure_python_weighted_npairs_xy import pure_python_weighted_npairs_xy
from .external_delta_sigma import external_delta_sigma

from ..new_delta_sigma import new_delta_sigma
from ..surface_density import surface_density_in_annulus, surface_density_in_cylinder
from ..surface_density_helpers import log_interpolation_with_inner_zero_masking as log_interp

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_thin_cylindrical_shell_of_points

__all__ = ('test_delta_sigma_consistency', )

fixed_seed = 43


def test_delta_sigma_consistency():
    """This testing function guarantees consistency between the new_delta_sigma
    function and the surface_density_in_annulus and surface_density_in_cylinder functions,
    effectively freezing the internal calculation of new_delta_sigma.
    """
    num_centers, num_ptcl = 100, 500
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 5)
    Lbox = 1.

    rp_mids, ds = new_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)

    sigma_annulus = surface_density_in_annulus(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)
    sigma_inside_cylinder = surface_density_in_cylinder(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)

    sigma_inside_cylinder_interp = log_interp(sigma_inside_cylinder, rp_bins, rp_mids)
    implied_delta_sigma = sigma_inside_cylinder_interp - sigma_annulus
    assert np.allclose(implied_delta_sigma, ds, rtol=0.001)
