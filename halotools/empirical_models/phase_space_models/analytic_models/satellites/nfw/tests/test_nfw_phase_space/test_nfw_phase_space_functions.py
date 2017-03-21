"""
"""
import numpy as np

from ...nfw_phase_space import NFWPhaseSpace
from ...kernels.unbiased_isotropic_velocity import (
    dimensionless_radial_velocity_dispersion as dimless_vrad_disp_kernel)

from ........mock_observables import relative_positions_and_velocities

__all__ = ['test_mc_consistency1']

fixed_seed = 43


def test_mc_consistency1():
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    num_gals = int(10)
    t = nfw.mc_generate_nfw_phase_space_points(Ngals=num_gals, seed=43)

    radial_dist = np.sqrt(t['x']**2 + t['y']**2 + t['z']**2)
    assert np.allclose(radial_dist, t['radial_position'])


def test_mc_consistency2():
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    num_gals = int(10)
    t = nfw.mc_generate_nfw_phase_space_points(Ngals=num_gals, seed=43)

    xrel, vxrel = relative_positions_and_velocities(t['x'], 0, v1=t['vx'], v2=0)
    yrel, vyrel = relative_positions_and_velocities(t['y'], 0, v1=t['vy'], v2=0)
    zrel, vzrel = relative_positions_and_velocities(t['z'], 0, v1=t['vz'], v2=0)

    radial_dist = np.sqrt(t['x']**2 + t['y']**2 + t['z']**2)
    vrad = (xrel*vxrel + yrel*vyrel + zrel*vzrel)/radial_dist
    assert np.allclose(vrad, t['radial_velocity'])


def test_dimensionless_radial_velocity_dispersion1():
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    scaled_radius, conc = 0.5, 5
    kernel_vrad_disp = dimless_vrad_disp_kernel(scaled_radius, conc)
    class_vrad_disp = nfw.dimensionless_radial_velocity_dispersion(scaled_radius, conc)
    assert class_vrad_disp == kernel_vrad_disp


def test_dimensionless_radial_velocity_dispersion2():
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    scaled_radius, conc = np.logspace(-2, 0, 10), 5
    kernel_vrad_disp = dimless_vrad_disp_kernel(scaled_radius, conc)
    class_vrad_disp = nfw.dimensionless_radial_velocity_dispersion(scaled_radius, conc)
    assert np.all(class_vrad_disp == kernel_vrad_disp)


def test_radial_velocity_dispersion1():
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    conc, mass = 5, 1e12
    rvir = nfw.halo_mass_to_halo_radius(mass)
    scaled_radius = 0.5
    physical_radius = scaled_radius*rvir
    kernel_vrad_disp = dimless_vrad_disp_kernel(scaled_radius, conc)
    class_vrad_disp = nfw.radial_velocity_dispersion(physical_radius, mass, conc)/nfw.virial_velocity(mass)
    assert class_vrad_disp == kernel_vrad_disp


def test_radial_velocity_dispersion2():
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    conc, mass = 5, 1e12
    rvir = nfw.halo_mass_to_halo_radius(mass)
    scaled_radius = np.logspace(-2, 0, 10)
    physical_radius = scaled_radius*rvir

    kernel_vrad_disp = dimless_vrad_disp_kernel(scaled_radius, conc)
    class_vrad_disp = nfw.radial_velocity_dispersion(physical_radius, mass, conc)/nfw.virial_velocity(mass)
    assert np.allclose(class_vrad_disp, kernel_vrad_disp)


def test_vrad_disp_from_lookup():
    r""" Method used to test `~halotools.empirical_models.NFWPhaseSpace._vrad_disp_from_lookup`.

    Method verifies that all scaled velocities are between zero and unity.

    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    c15 = np.zeros(10)
    scaled_radius = np.logspace(-2, 0, len(c15))
    vr_disp = nfw._vrad_disp_from_lookup(scaled_radius, c15, seed=43)

    assert np.all(vr_disp < 1)
    assert np.all(vr_disp > 0)

