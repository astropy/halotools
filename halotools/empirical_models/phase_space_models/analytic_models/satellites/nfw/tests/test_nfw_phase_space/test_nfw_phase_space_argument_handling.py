"""
"""
import numpy as np

from ...nfw_phase_space import NFWPhaseSpace


__all__ = ('test_mc_radial_velocity_float_vs_array_args1', )

fixed_seed = 43


def test_mc_radial_velocity_float_vs_array_args1():
    """ Calling mc_radial_velocity with a float for concentration vs.
    an array of identical values should produce the same result.

    In this version of the test,
    all the array-versions of the arguments have shape (1, ).

    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    conc = 10
    mass = 1e12
    scaled_radius = 0.4
    scaled_radius_array = np.atleast_1d(scaled_radius)
    concarr = np.atleast_1d(conc)

    mc_vr_from_arr = nfw.mc_radial_velocity(scaled_radius_array, mass, concarr, seed=43)
    mc_vr_from_float = nfw.mc_radial_velocity(scaled_radius, mass, conc, seed=43)
    assert np.shape(mc_vr_from_arr) == np.shape(mc_vr_from_float)
    assert np.allclose(mc_vr_from_arr, mc_vr_from_float)


def test_mc_radial_velocity_float_vs_array_args2():
    """ Calling mc_radial_velocity with a float for concentration vs.
    an array of identical values should produce the same result.

    In this version of the test,
    ``scaled_radius`` and ``concarr`` have shape (100, ) but ``mass`` is a float.

    """
    npts = 100
    conc = 10
    mass = 1e12
    scaled_radius = 0.4
    scaled_radius_array = np.zeros(npts) + scaled_radius
    concarr = np.zeros_like(scaled_radius_array) + conc

    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    mc_vr_from_arr = nfw.mc_radial_velocity(scaled_radius_array, mass, concarr, seed=43)
    mc_vr_from_float = nfw.mc_radial_velocity(scaled_radius_array, mass, conc, seed=43)
    assert mc_vr_from_arr.shape == mc_vr_from_float.shape
    assert np.allclose(mc_vr_from_arr, mc_vr_from_float)


def test_mc_radial_velocity_float_vs_array_args3():
    """ Calling mc_radial_velocity with a float for concentration vs.
    an array of identical values should produce the same result.

    In this version of the test,
    ``scaled_radius``, ``concarr`` and ``mass`` all have shape (100, ).

    """
    npts = 100
    conc = 10
    mass = 1e12
    scaled_radius = 0.4
    scaled_radius_array = np.zeros(npts) + scaled_radius
    concarr = np.zeros_like(scaled_radius_array) + conc
    massarr = np.zeros_like(scaled_radius_array) + mass

    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    mc_vr_from_arr = nfw.mc_radial_velocity(scaled_radius_array, massarr, concarr, seed=43)
    mc_vr_from_float = nfw.mc_radial_velocity(scaled_radius_array, mass, conc, seed=43)
    assert mc_vr_from_arr.shape == mc_vr_from_float.shape
    assert np.allclose(mc_vr_from_arr, mc_vr_from_float)


def test_mc_halo_centric_pos_float_vs_array_args():
    """ Calling mc_halo_centric_pos with a float for halo_radius vs.
    an array of identical values should produce the same result.

    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    npts = 100
    conc = 10.
    mass = 1e12
    scaled_radius = 0.4
    scaled_radius_array = np.zeros(npts) + scaled_radius
    concarr = np.zeros_like(scaled_radius_array) + conc
    rvir = nfw.halo_mass_to_halo_radius(mass)
    rvir_array = np.zeros_like(concarr) + rvir

    mc_r_from_arr = nfw.mc_halo_centric_pos(concarr, halo_radius=rvir_array, seed=43)
    mc_r_from_mixed = nfw.mc_halo_centric_pos(concarr, halo_radius=rvir, seed=43)

    assert np.allclose(mc_r_from_arr, mc_r_from_mixed)


def test_mc_halo_centric_pos_float_arg():
    """ When calling the mc_halo_centric_pos method with float arguments,
    ensure the returned result has the correct shape
    """
    nfw = NFWPhaseSpace(concentration_bins=np.array((5, 10, 15)))

    conc = 10.
    mass = 1e12
    rvir = nfw.halo_mass_to_halo_radius(mass)

    mc_r_from_float = nfw.mc_halo_centric_pos(conc, halo_radius=rvir, seed=43)
    assert np.shape(mc_r_from_float) == (3, 1)
