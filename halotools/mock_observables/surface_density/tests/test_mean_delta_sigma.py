"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from astropy.cosmology import Planck15
import pytest

from .external_delta_sigma import external_delta_sigma

from ..mean_delta_sigma import mean_delta_sigma
from ..surface_density import surface_density_in_annulus, surface_density_in_cylinder
from ..surface_density_helpers import log_interpolation_with_inner_zero_masking as log_interp

from ..mean_delta_sigma_one_two_halo_decomp import mean_delta_sigma_one_two_halo_decomp

from ....empirical_models import PrebuiltSubhaloModelFactory
from ....sim_manager import CachedHaloCatalog
from ....mock_observables import return_xyz_formatted_array

__all__ = ('test_mean_delta_sigma1', )

fixed_seed = 43


# @pytest.mark.slow
# def test_mean_delta_sigma1():
#     """Enforce approximate agreement with scipy-based result
#     """
#     model = PrebuiltSubhaloModelFactory('behroozi10')
#     try:
#         halocat = CachedHaloCatalog()
#     except:
#         return  #  Skip test if the environment does not have the default halo catalog
#     model.populate_mock(halocat, seed=fixed_seed)

#     px = model.mock.ptcl_table['x']
#     py = model.mock.ptcl_table['y']
#     pz = model.mock.ptcl_table['z']
#     Nptcls_to_keep = int(1e5)
#     randomizer = np.random.random(len(model.mock.ptcl_table))
#     sorted_randoms = np.sort(randomizer)
#     ptcl_mask = np.where(sorted_randoms < sorted_randoms[Nptcls_to_keep])[0]
#     particles = return_xyz_formatted_array(px, py, pz, mask=ptcl_mask)

#     x = model.mock.galaxy_table['x']
#     y = model.mock.galaxy_table['y']
#     z = model.mock.galaxy_table['z']
#     mstar105_mask = (model.mock.galaxy_table['stellar_mass'] > 10**10.25)
#     mstar105_mask *= (model.mock.galaxy_table['stellar_mass'] < 10**10.5)
#     galaxies = return_xyz_formatted_array(x, y, z, mask=mstar105_mask)

#     period = halocat.Lbox[0]
#     projection_period = period
#     logrp_bins = np.linspace(0, np.log10(25), 15)
#     rp_bins = 10**logrp_bins
#     rp_mids = 10**(0.5*(logrp_bins[1:] + logrp_bins[:-1]))

#     rp_mids_external, dsigma_external = external_delta_sigma(galaxies[:, :2], particles[:, :2],
#         rp_bins, period, projection_period, cosmology=Planck15)

#     num_ptcl_per_dim = 2048
#     downsampling_factor = num_ptcl_per_dim**3/float(particles.shape[0])
#     dsigma = mean_delta_sigma(galaxies, particles, halocat.particle_mass,
#         downsampling_factor, rp_bins, halocat.Lbox)

#     dsigma_interpol = np.exp(np.interp(np.log(rp_mids_external),
#             np.log(rp_mids), np.log(dsigma)))

#     assert np.allclose(dsigma_interpol, dsigma_external, rtol=0.2)


@pytest.mark.xfail
def test_delta_sigma_consistency():
    """This testing function enforces consistency between the mean_delta_sigma
    function and the surface_density_in_annulus and surface_density_in_cylinder functions.
    This was intended to freeze the internal calculation of delta_sigma in <=v0.6,
    which is now obselete, and so this test may be deleted.
    """
    num_centers, num_ptcl = 100, 500
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 5)
    rp_mids = (0.5*(rp_bins[1:] + rp_bins[:-1]))
    Lbox = 1.

    ds = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)

    sigma_annulus = surface_density_in_annulus(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)
    sigma_inside_cylinder = surface_density_in_cylinder(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)

    sigma_inside_cylinder_interp = log_interp(sigma_inside_cylinder, rp_bins, rp_mids)
    implied_delta_sigma = sigma_inside_cylinder_interp - sigma_annulus
    assert np.allclose(implied_delta_sigma, ds, rtol=0.1)


def test_delta_sigma_consistency2():
    """Enforce agreement between parallel and serial,
    with and without per-object options.
    """
    num_centers, num_ptcl = 100, 500
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 5)
    Lbox = 1.

    ds0 = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox, num_threads=2)
    ds1 = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)
    ds2 = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox, per_object=True)
    ds3 = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox, per_object=True, num_threads=2)
    assert np.allclose(ds0, ds1)
    assert np.allclose(ds2, ds3)
    assert np.allclose(ds1, np.mean(ds2, axis=0))


def test_delta_sigma_consistency3():
    """Enforce agreement between PBC and no-PBC when all points are at the
    center of the box.
    """
    num_centers, num_ptcl = 100, 500
    with NumpyRNGContext(fixed_seed):
        centers = np.random.uniform(0.499, 0.501, size=(num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 5)
    Lbox = 1.

    ds0 = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, period=Lbox)
    ds1 = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, period=None)

    assert np.allclose(ds0, ds1)


def test_delta_sigma_consistency4():
    """Enforce agreement between 1-2-halo decomposition and regular function
    """
    num_centers, num_ptcl = 10, 500000
    galaxy_halo_ids = np.arange(num_centers)
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))
        particle_halo_ids = np.random.randint(0, num_centers, num_ptcl)

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 15)
    Lbox = 1.

    ds0 = mean_delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, period=Lbox)

    ds_1h, ds_2h = mean_delta_sigma_one_two_halo_decomp(centers, particles, particle_masses,
        downsampling_factor, rp_bins, period=Lbox,
        galaxy_halo_ids=galaxy_halo_ids, particle_halo_ids=particle_halo_ids)

    ds_1h_parallel, ds_2h_parallel = mean_delta_sigma_one_two_halo_decomp(centers, particles, particle_masses,
        downsampling_factor, rp_bins, period=Lbox,
        galaxy_halo_ids=galaxy_halo_ids, particle_halo_ids=particle_halo_ids, num_threads=2)

    assert np.allclose(ds0, ds_1h+ds_2h)

    assert np.allclose(ds_1h, ds_1h_parallel)
    assert np.allclose(ds_2h, ds_2h_parallel)


def test_delta_sigma_consistency5():
    """Enforce agreement between 1-2-halo decomposition and regular function
    """
    num_centers, num_ptcl = 10, 500000
    galaxy_halo_ids = np.arange(num_centers)
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))
        particle_halo_ids = np.random.randint(0, num_centers, num_ptcl)

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 15)
    Lbox = 1.

    ds_1h_per_obj, ds_2h_per_obj = mean_delta_sigma_one_two_halo_decomp(
        centers, particles, particle_masses,
        downsampling_factor, rp_bins, period=Lbox,
        galaxy_halo_ids=galaxy_halo_ids, particle_halo_ids=particle_halo_ids)

    ds_1h_per_obj_parallel, ds_2h_per_obj_parallel = mean_delta_sigma_one_two_halo_decomp(
        centers, particles, particle_masses,
        downsampling_factor, rp_bins, period=Lbox,
        galaxy_halo_ids=galaxy_halo_ids, particle_halo_ids=particle_halo_ids, num_threads=3)

    assert np.allclose(ds_1h_per_obj, ds_1h_per_obj_parallel)
    assert np.allclose(ds_2h_per_obj, ds_2h_per_obj_parallel)


def test_delta_sigma_raises_exceptions1():
    num_centers, num_ptcl = 100, 500
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl-1)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 5)
    Lbox = 1.

    with pytest.raises(AssertionError) as err:
        ds = mean_delta_sigma(centers, particles, particle_masses,
            downsampling_factor, rp_bins, Lbox)
    substr = "Must have same number of ``particle_masses`` as particles"
    assert substr in err.value.args[0]


def test_delta_sigma_raises_exceptions2():
    num_centers, num_ptcl = 100, 500
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 0.5

    rp_bins = np.linspace(0.1, 0.3, 5)
    Lbox = 1.

    with pytest.raises(AssertionError) as err:
        ds = mean_delta_sigma(centers, particles, particle_masses,
            downsampling_factor, rp_bins, Lbox)
    substr = "downsampling_factor = 0.5 < 1, which is impossible".format(downsampling_factor)
    assert substr in err.value.args[0]
