"""
"""
import numpy as np

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace


__all__ = ('test_mc_radial_velocity_float_vs_array_args1', )

fixed_seed = 43

conc_bins = np.linspace(5, 10, 3)
gal_bias_bins = np.linspace(0.1, 20, 2)
gal_bias_bins = np.insert(gal_bias_bins, np.searchsorted(gal_bias_bins, 1), 1)


def test_mc_radial_velocity_float_vs_array_args1():
    """ Calling mc_radial_velocity with a float for concentration vs.
    an array of identical values should produce the same result.

    In this version of the test,
    all the array-versions of the arguments have shape (1, ).

    """
    nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins, conc_gal_bias_bins=gal_bias_bins)

    conc = 10
    mass = 1e12
    scaled_radius = 0.4
    scaled_radius_array = np.atleast_1d(scaled_radius)
    concarr = np.atleast_1d(conc)
    galbias = 1.
    galbiasarr = np.atleast_1d(galbias)

    mc_vr_from_arr = nfw.mc_radial_velocity(scaled_radius_array, mass, concarr, galbiasarr, seed=43)
    mc_vr_from_float = nfw.mc_radial_velocity(scaled_radius, mass, conc, galbias, seed=43)
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
    galbias = 1.
    galbiasarr = np.zeros_like(scaled_radius_array) + galbias

    nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins, conc_gal_bias_bins=gal_bias_bins)

    mc_vr_from_arr = nfw.mc_radial_velocity(scaled_radius_array, mass, concarr, galbiasarr, seed=43)
    mc_vr_from_float = nfw.mc_radial_velocity(scaled_radius_array, mass, conc, galbias, seed=43)
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
    galbias = 1.
    galbiasarr = np.zeros_like(scaled_radius_array) + galbias

    nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins, conc_gal_bias_bins=gal_bias_bins)

    mc_vr_from_arr = nfw.mc_radial_velocity(scaled_radius_array, massarr, concarr, galbiasarr, seed=43)
    mc_vr_from_float = nfw.mc_radial_velocity(scaled_radius_array, mass, conc, galbias, seed=43)
    assert mc_vr_from_arr.shape == mc_vr_from_float.shape
    assert np.allclose(mc_vr_from_arr, mc_vr_from_float)


def test_mc_halo_centric_pos_float_vs_array_args():
    """ Calling mc_halo_centric_pos with a float for halo_radius vs.
    an array of identical values should produce the same result.

    """
    nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins, conc_gal_bias_bins=gal_bias_bins)

    npts = 100
    conc = 10.
    mass = 1e12
    scaled_radius = 0.4
    scaled_radius_array = np.zeros(npts) + scaled_radius
    concarr = np.zeros_like(scaled_radius_array) + conc
    rvir = nfw.halo_mass_to_halo_radius(mass)
    rvir_array = np.zeros_like(concarr) + rvir
    galbias = 1.
    galbiasarr = np.zeros_like(scaled_radius_array) + galbias

    mc_r_from_arr = nfw.mc_halo_centric_pos(concarr, galbiasarr, halo_radius=rvir_array, seed=43)
    mc_r_from_mixed = nfw.mc_halo_centric_pos(concarr, galbiasarr, halo_radius=rvir, seed=43)

    assert np.allclose(mc_r_from_arr, mc_r_from_mixed)


def test_mc_halo_centric_pos_float_arg():
    """ When calling the mc_halo_centric_pos method with float arguments,
    ensure the returned result has the correct shape
    """
    nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins, conc_gal_bias_bins=gal_bias_bins)

    conc = 10.
    mass = 1e12
    rvir = nfw.halo_mass_to_halo_radius(mass)
    galbias = 1.

    mc_r_from_float = nfw.mc_halo_centric_pos(conc, galbias, halo_radius=rvir, seed=43)
    assert np.shape(mc_r_from_float) == (3, 1)
