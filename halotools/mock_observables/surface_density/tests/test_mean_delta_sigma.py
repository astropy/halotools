"""
"""
import numpy as np
from ..mean_delta_sigma import mean_delta_sigma


def test_mean_delta_sigma_returns_correct_shape_serial():
    ngals, nparts = 500, 5000
    nbins = 30
    Lbox = 500
    rp_bins = np.logspace(-1, 1.25, nbins)
    rng = np.random.RandomState(43)
    galaxies = Lbox*rng.uniform(size=(ngals, 3))
    particles = Lbox*rng.uniform(size=(nparts, 3))
    particle_masses = 1.
    downsampling_factor = 1.

    effective_particle_masses = particle_masses * downsampling_factor

    mean_ds = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox)
    assert mean_ds.shape == (nbins-1, )

    ds_per_object = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox,
        per_object=True)
    assert ds_per_object.shape == (ngals, nbins-1)


def test_mean_delta_sigma_returns_correct_shape_parallel():
    ngals, nparts = 500, 5000
    nbins = 30
    Lbox = 500
    rp_bins = np.logspace(-1, 1.25, nbins)
    rng = np.random.RandomState(43)
    galaxies = Lbox*rng.uniform(size=(ngals, 3))
    particles = Lbox*rng.uniform(size=(nparts, 3))
    particle_masses = 1.
    downsampling_factor = 1.
    effective_particle_masses = particle_masses * downsampling_factor

    mean_ds = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox,
        num_threads=2)
    assert mean_ds.shape == (nbins-1, )

    ds_per_object = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox,
        per_object=True, num_threads=2)
    assert ds_per_object.shape == (ngals, nbins-1)


def test_mean_delta_sigma_serial_parallel_agree():
    ngals, nparts = 500, 5000
    nbins = 30
    Lbox = 500
    rp_bins = np.logspace(-1, 1.25, nbins)
    rng = np.random.RandomState(43)
    galaxies = Lbox*rng.uniform(size=(ngals, 3))
    particles = Lbox*rng.uniform(size=(nparts, 3))
    particle_masses = 1.
    downsampling_factor = 1.
    effective_particle_masses = particle_masses * downsampling_factor

    mean_ds_serial = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox,
        num_threads=1)

    mean_ds_parallel = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox,
        num_threads=2)

    assert np.allclose(mean_ds_serial, mean_ds_parallel)


def test_delta_sigma_per_object_serial_parallel_agree():
    ngals, nparts = 500, 5000
    nbins = 30
    Lbox = 500
    rp_bins = np.logspace(-1, 1.25, nbins)
    rng = np.random.RandomState(43)
    galaxies = Lbox*rng.uniform(size=(ngals, 3))
    particles = Lbox*rng.uniform(size=(nparts, 3))
    particle_masses = 1.
    downsampling_factor = 1.
    effective_particle_masses = particle_masses * downsampling_factor

    ds_per_obj_serial = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox,
        num_threads=1, per_object=True)

    ds_per_obj_parallel = mean_delta_sigma(
        galaxies, particles, effective_particle_masses, rp_bins, period=Lbox,
        num_threads=2, per_object=True)

    assert np.allclose(ds_per_obj_serial, ds_per_obj_parallel)
