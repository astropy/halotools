"""
"""
import numpy as np
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext
import pytest

from ...nfw_phase_space import NFWPhaseSpace


__all__ = ('test_mc_unit_sphere_stochasticity', )

fixed_seed = 43


def get_dummy_halo_table(npts):
    x = np.linspace(-1, 1, npts)
    c = np.zeros(npts) + 5.
    m = np.zeros(npts) + 1e12
    zeros = np.zeros(npts)

    return Table({'halo_x': x, 'halo_y': zeros+0.25, 'halo_z': zeros+0.5,
        'host_centric_distance': x, 'halo_rvir': 3*x, 'conc_NFWmodel': c,
        'halo_vx': zeros, 'halo_vy': zeros, 'halo_vz': zeros, 'halo_mvir': m,
        'x': zeros, 'y': zeros, 'z': zeros, 'vx': zeros, 'vy': zeros, 'vz': zeros})


def test_mc_unit_sphere_stochasticity():
    r""" Method used to test correctness of stochasticity/deterministic behavior of
    `~halotools.empirical_models.NFWPhaseSpace.mc_unit_sphere`.
    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((2, 5, 10)))
    x1, y1, z1 = nfw.mc_unit_sphere(100, seed=43)
    x2, y2, z2 = nfw.mc_unit_sphere(100, seed=43)
    x3, y3, z3 = nfw.mc_unit_sphere(100, seed=None)
    assert np.allclose(x1, x2, rtol=0.001)
    assert not np.allclose(x1, x3, rtol=0.001)


def test_mc_unit_sphere():
    r""" Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_unit_sphere`.

    This test verifies that all returned 3d points are at unit distance from the origin.
    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((2, 5, 10)))
    x, y, z = nfw.mc_unit_sphere(100, seed=43)
    pos = np.vstack([x, y, z]).T
    norm = np.linalg.norm(pos, axis=1)
    assert np.allclose(norm, 1, rtol=1e-4)


def test_mc_dimensionless_radial_distance_stochasticity():
    r""" Method used to test correctness of stochasticity/deterministic behavior of
    `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`.
    """
    Npts = int(100)
    c5 = np.zeros(Npts) + 5

    nfw = NFWPhaseSpace(concentration_bins=np.array((2, 5, 10)))
    x1 = nfw._mc_dimensionless_radial_distance(c5, seed=43)
    x2 = nfw._mc_dimensionless_radial_distance(c5, seed=43)
    x3 = nfw._mc_dimensionless_radial_distance(c5, seed=None)
    assert np.allclose(x1, x2, rtol=0.001)
    assert not np.allclose(x1, x3, rtol=0.001)


def test_mc_solid_sphere():
    r""" Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_solid_sphere`.

    Method ensures that all returned points lie inside the unit sphere.
    """
    Npts = int(100)
    c5 = np.zeros(Npts) + 5

    nfw = NFWPhaseSpace(concentration_bins=np.array((2, 5, 10)))
    x, y, z = nfw.mc_solid_sphere(c5, seed=43)
    pos = np.vstack([x, y, z]).T
    norm = np.linalg.norm(pos, axis=1)
    assert np.all(norm < 1)
    assert np.all(norm > 0)
    assert np.all(x > -1)
    assert np.all(x < 1)
    assert np.all(y > -1)
    assert np.all(y < 1)
    assert np.all(z > -1)
    assert np.all(z < 1)


def test_mc_solid_sphere_stochasticity():
    r""" Method used to test correctness of stochasticity/deterministic behavior of
    `~halotools.empirical_models.NFWPhaseSpace.mc_solid_sphere`.
    """
    Npts = int(100)
    c5 = np.zeros(Npts) + 5

    nfw = NFWPhaseSpace(concentration_bins=np.array((2, 5, 10)))
    x1, y1, z1 = nfw.mc_solid_sphere(c5, seed=43)
    x2, y2, z2 = nfw.mc_solid_sphere(c5, seed=43)
    x3, y3, z3 = nfw.mc_solid_sphere(c5, seed=None)

    assert np.allclose(x1, x2, rtol=0.001)
    assert not np.allclose(x1, x3, rtol=0.001)


def test_mc_halo_centric_pos():
    r""" Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`.

    Method verifies

    1. All returned points lie within the correct radial distance

    2. Increasing the input concentration decreases the mean and median radial distance of the returned points.

    """
    r = 0.25
    Npts = int(100)
    c5 = np.zeros(Npts) + 5
    c10 = np.zeros(Npts) + 10
    c15 = np.zeros(Npts) + 15
    halo_radius = np.zeros(len(c5)) + r

    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    x15, y15, z15 = nfw.mc_halo_centric_pos(c15,
        halo_radius=halo_radius, seed=43)
    assert np.all(x15 > -r)
    assert np.all(x15 < r)
    assert np.all(y15 > -r)
    assert np.all(y15 < r)
    assert np.all(z15 > -r)
    assert np.all(z15 < r)

    pos15 = np.vstack([x15, y15, z15]).T
    norm15 = np.linalg.norm(pos15, axis=1)
    assert np.all(norm15 < r)
    assert np.all(norm15 > 0)

    x5, y5, z5 = nfw.mc_halo_centric_pos(c5,
        halo_radius=halo_radius, seed=43)
    pos5 = np.vstack([x5, y5, z5]).T
    norm5 = np.linalg.norm(pos5, axis=1)

    x10, y10, z10 = nfw.mc_halo_centric_pos(c10,
        halo_radius=halo_radius,  seed=43)
    pos10 = np.vstack([x10, y10, z10]).T
    norm10 = np.linalg.norm(pos10, axis=1)

    assert np.mean(norm5) > np.mean(norm10)
    assert np.mean(norm10) > np.mean(norm15)

    assert np.median(norm5) > np.median(norm10)
    assert np.median(norm10) > np.median(norm15)

    x10a, y10a, z10a = nfw.mc_halo_centric_pos(c10,
        halo_radius=halo_radius*2, seed=43)
    pos10a = np.vstack([x10a, y10a, z10a]).T
    norm10a = np.linalg.norm(pos10a, axis=1)

    assert np.any(norm10a > r)
    assert np.all(norm10a < 2*r)


def test_mc_halo_centric_pos_stochasticity():
    r""" Method used to test stochasticity/deterministic behavior of
    `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`.

    """
    r = 0.25
    Npts = int(100)
    c15 = np.zeros(Npts) + 15
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    halo_radius = np.zeros(len(c15)) + r
    x15a, y15a, z15a = nfw.mc_halo_centric_pos(c15,
        halo_radius=halo_radius, seed=43)
    x15b, y15b, z15b = nfw.mc_halo_centric_pos(c15,
        halo_radius=halo_radius, seed=43)
    x15c, y15c, z15c = nfw.mc_halo_centric_pos(c15,
        halo_radius=halo_radius, seed=None)
    assert np.allclose(x15a, x15b, rtol=0.01)
    assert not np.allclose(x15a, x15c, rtol=0.01)


def test_mc_pos():
    r""" Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`.

    Method verifies that passing an input ``seed`` results in deterministic behavior.

    """
    r = 0.25
    Npts = int(100)
    c15 = np.zeros(Npts) + 15
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))
    halo_radius = np.zeros(len(c15)) + r
    x1, y1, z1 = nfw.mc_pos(c15,
        halo_radius=halo_radius, seed=43)
    x2, y2, z2 = nfw.mc_halo_centric_pos(c15,
        halo_radius=halo_radius, seed=43)
    assert np.all(x1 == x2)
    assert np.all(y1 == y2)
    assert np.all(z1 == z2)

    nfw.mc_pos(table=get_dummy_halo_table(Npts))


def test_mc_radial_velocity_consistency():
    r""" Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_radial_velocity`.

    Method generates a Monte Carlo velocity profile realization with all points at
    a specific radius and compares the manually computed velocity dispersion
    to the analytical expectation.
    """
    npts = int(1e4)
    conc = 10
    mass = 1e12
    scaled_radius = 0.4
    scaled_radius_array = np.zeros(npts) + scaled_radius
    concarr = np.zeros_like(scaled_radius_array) + conc

    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    mc_vr = nfw.mc_radial_velocity(scaled_radius_array, mass, concarr, seed=43)
    vr_dispersion_from_monte_carlo = np.std(mc_vr)

    rvir = nfw.halo_mass_to_halo_radius(mass)
    radius = scaled_radius*rvir
    analytical_result = nfw.radial_velocity_dispersion(radius, mass, conc)

    assert np.allclose(vr_dispersion_from_monte_carlo, analytical_result, rtol=0.05)


def test_mc_radial_velocity_stochasticity():
    r""" Method used to verify correct deterministic/stochastic behavior of
    `~halotools.empirical_models.NFWPhaseSpace.mc_radial_velocity`.

    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    npts = int(100)
    conc = 10
    carr = np.zeros(npts) + conc

    mass = 1e12
    rmax = nfw.rmax(mass, conc)
    r = np.zeros(npts) + rmax
    rvir = nfw.halo_mass_to_halo_radius(mass)
    scaled_radius = r/rvir

    mc_vr_seed43a = nfw.mc_radial_velocity(scaled_radius, mass, carr, seed=43)
    mc_vr_seed43b = nfw.mc_radial_velocity(scaled_radius, mass, carr, seed=43)
    mc_vr_seed44 = nfw.mc_radial_velocity(scaled_radius, mass, carr, seed=44)

    assert np.allclose(mc_vr_seed43a, mc_vr_seed43b, rtol=0.001)
    assert not np.allclose(mc_vr_seed43a, mc_vr_seed44, rtol=0.001)


def test_mc_pos1():
    r""" Verify that the seed keyword is treated properly.
    This function serves as a regression test for https://github.com/astropy/halotools/issues/672.
    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))
    halos = Table()
    num_halos = 200
    halos['x'] = np.zeros(num_halos)
    halos['y'] = np.zeros(num_halos)
    halos['z'] = np.zeros(num_halos)
    halos['host_centric_distance'] = 0.
    halos['halo_rvir'] = 1.
    halos['conc_NFWmodel'] = 5.
    halos['halo_mvir'] = 1e12
    nfw.mc_pos(table=halos, seed=43)

    assert np.min(halos['x']) < -0.7
    assert np.min(halos['y']) < -0.7
    assert np.min(halos['z']) < -0.7
    assert np.max(halos['x']) > 0.7
    assert np.max(halos['y']) > 0.7
    assert np.max(halos['z']) > 0.7


def test_mc_vel1():
    r"""
    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    halo_table = get_dummy_halo_table(10)
    assert np.all(halo_table['vx'] == halo_table['halo_vx'])

    nfw.mc_vel(halo_table, seed=fixed_seed)
    assert np.any(halo_table['vx'] != halo_table['halo_vx'])


def test_mc_vel2():
    r""" Method verifies that seed keyword is treated properly.
    This serves as a regression test for https://github.com/astropy/halotools/issues/672
    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))
    halos = Table()
    num_halos = 100
    halos['vx'] = np.zeros(num_halos)
    halos['vy'] = np.zeros(num_halos)
    halos['vz'] = np.zeros(num_halos)
    halos['host_centric_distance'] = np.logspace(-2, 0, num_halos)
    halos['halo_rvir'] = 1.
    halos['conc_NFWmodel'] = 5.
    halos['halo_mvir'] = 1e12
    nfw.mc_vel(halos, seed=43)
    assert not np.all(halos['vx'] == halos['vy'])
    assert not np.all(halos['vx'] == halos['vz'])


def test_seed_treatment1():
    r""" Regression test for https://github.com/astropy/halotools/issues/672.
    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))
    satellites = nfw.mc_generate_nfw_phase_space_points(seed=43, Ngals=10)
    assert np.any(satellites['vx'] != satellites['vy'])


def test_automatic_ngal_inference1():
    """ Regression test for Issue #870 https://github.com/astropy/halotools/issues/870
    """
    nfw = NFWPhaseSpace()
    posvel_table = nfw.mc_generate_nfw_phase_space_points(mass=(1e12, 1e13))
    posvel_table = nfw.mc_generate_nfw_phase_space_points(conc=(5, 13))
    posvel_table = nfw.mc_generate_nfw_phase_space_points(conc=(5, 13), mass=(1e12, 1e13))
    posvel_table = nfw.mc_generate_nfw_phase_space_points(conc=5, mass=1e13, Ngals=100)

    with pytest.raises(AssertionError) as err:
        posvel_table = nfw.mc_generate_nfw_phase_space_points(
            conc=(5, 13, 6), mass=(1e12, 1e13))
    substr = "Input ``mass`` and ``conc`` must have same length"
    assert substr in err.value.args[0]
