"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .pure_python_weighted_npairs_xy import pure_python_weighted_npairs_xy

from ..mass_in_cylinders import total_mass_enclosed_in_stack_of_cylinders as mass_in_cylinder_stack
from ..mass_in_cylinders import total_mass_enclosed_per_cylinder as mass_in_each_cylinder
from ...tests.cf_helpers import generate_3d_regular_mesh, generate_thin_cylindrical_shell_of_points

__all__ = ('test_mass_in_cylinder_stack_grid_of_points', )

fixed_seed = 43


def test_mass_in_cylinder_stack_grid_of_points():

    Lbox = 1.
    num_cyl_per_dim = 5
    centers = generate_3d_regular_mesh(num_cyl_per_dim)  #  0.1, 0.3, 0.5, 0.7, 0.9

    num_ptcl = 333
    xc, yc, zc = 0.3, 0.3, 0.3
    particles = generate_thin_cylindrical_shell_of_points(num_ptcl, 0.01, Lbox/2.,
        xc, yc, zc, seed=fixed_seed, Lbox=Lbox)

    masses = np.logspace(2, 5, particles.shape[0])
    downsampling_factor = 1.
    rp_bins = np.array((0.005, 0.02, 0.31))

    mass_encl = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor, rp_bins, Lbox)

    correct_result_inner_bin = 0.0
    assert np.allclose(correct_result_inner_bin, mass_encl[0])

    correct_result_middle_bin = num_cyl_per_dim*masses.sum()
    assert np.allclose(correct_result_middle_bin, mass_encl[1])


def test_mass_in_cylinder_stack_brute_force():
    """
    """
    Lbox = 1
    num_cyl, num_ptcl = 121, 231

    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_cyl, 3))
        particles = np.random.random((num_ptcl, 3))
        masses = np.random.rand(num_ptcl)

    downsampling_factor = 1.
    rp_bins = np.linspace(0.1, 0.3, 10)

    mass_encl = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor, rp_bins, Lbox)

    xarr1, yarr1 = centers[:, 0], centers[:, 1]
    xarr2, yarr2 = particles[:, 0], particles[:, 1]
    w2 = masses
    xperiod, yperiod = Lbox, Lbox
    counts, weighted_counts = pure_python_weighted_npairs_xy(
            xarr1, yarr1, xarr2, yarr2, w2, rp_bins, xperiod, yperiod)

    assert np.allclose(weighted_counts, mass_encl)


def test_mass_in_cylinder_stack_downsampling_factor():
    """
    """
    Lbox = 1
    num_cyl, num_ptcl = 124, 234

    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_cyl, 3))
        particles = np.random.random((num_ptcl, 3))
        masses = np.random.rand(num_ptcl)
    rp_bins = np.linspace(0.1, 0.3, 10)

    downsampling_factor1 = 3.
    mass_encl1 = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor1, rp_bins, Lbox)
    downsampling_factor2 = 25.
    mass_encl2 = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor2, rp_bins, Lbox)
    assert np.allclose(mass_encl2, mass_encl1*downsampling_factor2/downsampling_factor1)


def test_mass_in_cylinder_stack_parallel():
    """
    """
    Lbox = 1
    num_cyl, num_ptcl = 110, 239

    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_cyl, 3))
        particles = np.random.random((num_ptcl, 3))
        masses = np.random.rand(num_ptcl)

    downsampling_factor = 1.
    rp_bins = np.linspace(0.1, 0.3, 10)

    mass_encl_serial = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor, rp_bins, Lbox, num_threads=1)
    mass_encl_two_threads = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor, rp_bins, Lbox, num_threads=2)
    mass_encl_eleven_threads = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor, rp_bins, Lbox, num_threads='max')
    assert np.allclose(mass_encl_serial, mass_encl_two_threads)
    assert np.allclose(mass_encl_serial, mass_encl_eleven_threads)


def test_mass_per_cylinder():
    Lbox = 1
    num_cyl, num_ptcl = 121, 231

    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_cyl, 3))
        particles = np.random.random((num_ptcl, 3))
        masses = np.random.rand(num_ptcl)

    downsampling_factor = 1.
    rp_bins = np.linspace(0.1, 0.3, 10)

    mass_encl_stack = mass_in_cylinder_stack(centers, particles, masses,
        downsampling_factor, rp_bins, Lbox)
    mass_encl_per_cylinder = mass_in_each_cylinder(centers, particles, masses,
        downsampling_factor, rp_bins, Lbox)
    assert np.allclose(np.sum(mass_encl_per_cylinder, axis=0), mass_encl_stack)

