"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from astropy.tests.helper import pytest

from .external_delta_sigma import external_delta_sigma

from ..delta_sigma import delta_sigma, delta_sigma_from_precomputed_pairs
from ..surface_density import surface_density_in_annulus, surface_density_in_cylinder
from ..surface_density_helpers import log_interpolation_with_inner_zero_masking as log_interp
from ..mass_in_cylinders import total_mass_enclosed_per_cylinder

from ....empirical_models import PrebuiltSubhaloModelFactory
from ....sim_manager import CachedHaloCatalog
from ....mock_observables import return_xyz_formatted_array

__all__ = ('test_delta_sigma_consistency', )

fixed_seed = 43


@pytest.mark.slow
def test_delta_sigma1():
    """
    """
    model = PrebuiltSubhaloModelFactory('behroozi10')
    try:
        halocat = CachedHaloCatalog()
    except:
        return  #  Skip test if the environment does not have the default halo catalog
    model.populate_mock(halocat, seed=fixed_seed)

    px = model.mock.ptcl_table['x']
    py = model.mock.ptcl_table['y']
    pz = model.mock.ptcl_table['z']
    Nptcls_to_keep = int(1e5)
    randomizer = np.random.random(len(model.mock.ptcl_table))
    sorted_randoms = np.sort(randomizer)
    ptcl_mask = np.where(sorted_randoms < sorted_randoms[Nptcls_to_keep])[0]
    particles = return_xyz_formatted_array(px, py, pz, mask=ptcl_mask)

    x = model.mock.galaxy_table['x']
    y = model.mock.galaxy_table['y']
    z = model.mock.galaxy_table['z']
    mstar105_mask = (model.mock.galaxy_table['stellar_mass'] > 10**10.25)
    mstar105_mask *= (model.mock.galaxy_table['stellar_mass'] < 10**10.75)
    galaxies = return_xyz_formatted_array(x, y, z, mask=mstar105_mask)

    period = halocat.Lbox[0]
    projection_period = period
    rp_bins = np.logspace(np.log10(0.25), np.log10(15), 10)

    try:
        rp_mids_external, dsigma_external = external_delta_sigma(galaxies[:, :2], particles[:, :2],
            rp_bins, period, projection_period, cosmology=halocat.cosmology)
    except:
        return  #  skip test if testing environment has scipy version incompatibilities

    downsampling_factor = halocat.num_ptcl_per_dim**3/float(particles.shape[0])
    rp_mids, dsigma = delta_sigma(galaxies, particles, halocat.particle_mass,
        downsampling_factor, rp_bins, halocat.Lbox)

    dsigma_interpol = np.exp(np.interp(np.log(rp_mids_external),
            np.log(rp_mids), np.log(dsigma)))

    assert np.allclose(dsigma_interpol, dsigma_external, rtol=0.1)


def test_delta_sigma_consistency():
    """This testing function guarantees consistency between the delta_sigma
    function and the surface_density_in_annulus and surface_density_in_cylinder functions,
    effectively freezing the internal calculation of delta_sigma.
    """
    num_centers, num_ptcl = 100, 500
    with NumpyRNGContext(fixed_seed):
        centers = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 5)
    Lbox = 1.

    rp_mids, ds = delta_sigma(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)

    sigma_annulus = surface_density_in_annulus(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)
    sigma_inside_cylinder = surface_density_in_cylinder(centers, particles, particle_masses,
        downsampling_factor, rp_bins, Lbox)

    sigma_inside_cylinder_interp = log_interp(sigma_inside_cylinder, rp_bins, rp_mids)
    implied_delta_sigma = sigma_inside_cylinder_interp - sigma_annulus
    assert np.allclose(implied_delta_sigma, ds, rtol=0.001)


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
        rp_mids, ds = delta_sigma(centers, particles, particle_masses,
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
        rp_mids, ds = delta_sigma(centers, particles, particle_masses,
            downsampling_factor, rp_bins, Lbox)
    substr = "downsampling_factor = 0.5 < 1, which is impossible".format(downsampling_factor)
    assert substr in err.value.args[0]


def test_delta_sigma_from_precomputed_pairs():
    num_centers, num_ptcl = 1000, 5000
    with NumpyRNGContext(fixed_seed):
        galaxies = np.random.random((num_centers, 3))
        particles = np.random.random((num_ptcl, 3))

    particle_masses = np.ones(num_ptcl)
    downsampling_factor = 1

    rp_bins = np.linspace(0.1, 0.3, 5)
    period = 1.

    rp_mids, ds1 = delta_sigma(galaxies, particles, particle_masses,
        downsampling_factor, rp_bins, period)

    mass_encl = total_mass_enclosed_per_cylinder(galaxies, particles, particle_masses,
        downsampling_factor, rp_bins, period)
    rp_mids2, ds2 = delta_sigma_from_precomputed_pairs(galaxies, mass_encl, rp_bins, period)

    assert np.allclose(ds1, ds2)
