"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..surface_density_helpers import log_interpolation_with_inner_zero_masking as log_interp
from ..surface_density import surface_density_in_annulus, surface_density_in_cylinder

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_thin_cylindrical_shell_of_points

__all__ = ('test_surface_density_in_cylinder', )

fixed_seed = 43


def test_surface_density_in_cylinder():
    """ Given a tight locus of cylinders and a thin shell of particle surrounding them,
    verify that the surface_density_in_cylinder function gives the correct result
    for this analytically calculable case.
    """
    num_centers, num_ptcl = 10, 50
    xc, yc, zc = 0.1, 0.1, 0.1
    centers = generate_locus_of_3d_points(num_centers, xc=xc, yc=yc, zc=zc, seed=43)
    rp_shell = 0.05
    Lbox = 1
    particles = generate_thin_cylindrical_shell_of_points(num_ptcl, rp_shell, Lbox/2.,
        xc, yc, zc, seed=fixed_seed, Lbox=Lbox)
    with NumpyRNGContext(fixed_seed):
        particle_masses = np.random.rand(num_ptcl)

    downsampling_factor = 5.
    rp_bins = np.array((0.01, 2*rp_shell))

    sigma = surface_density_in_cylinder(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)
    assert sigma.shape[0] == rp_bins.shape[0]

    cylinder_areas = np.pi*rp_bins**2
    total_mass_encl = np.array((0, num_centers*downsampling_factor*particle_masses.sum()))
    assert np.allclose(sigma, total_mass_encl/(num_centers*cylinder_areas))


def test_surface_density_in_annulus():
    """ Given a tight locus of cylinders and a thin shell of particle surrounding them,
    verify that the surface_density_in_annulus
     function gives the correct result
    for this analytically calculable case.
    """
    num_centers, num_ptcl = 10, 50
    xc, yc, zc = 0.1, 0.1, 0.1
    centers = generate_locus_of_3d_points(num_centers, xc=xc, yc=yc, zc=zc, seed=43)
    rp_shell = 0.05
    Lbox = 1
    particles = generate_thin_cylindrical_shell_of_points(num_ptcl, rp_shell, Lbox/2.,
        xc, yc, zc, seed=fixed_seed, Lbox=Lbox)
    with NumpyRNGContext(fixed_seed):
        particle_masses = np.random.rand(num_ptcl)

    downsampling_factor = 5.
    rp_bins = np.array((0.01, 2*rp_shell, 3*rp_shell))

    sigma = surface_density_in_annulus(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)
    assert sigma.shape[0] == rp_bins.shape[0]-1

    annuli_areas = np.pi*np.diff(rp_bins**2)
    total_mass_encl = np.array((num_centers*downsampling_factor*particle_masses.sum(), 0))
    assert np.allclose(sigma, total_mass_encl/(num_centers*annuli_areas))
