#!/usr/bin/env python

import pytest
from unittest import TestCase

import numpy as np 
from astropy.table import Table 
from ...sim_manager import HaloCatalog
from ..phase_space_models import NFWPhaseSpace
from ...custom_exceptions import HalotoolsError

__all__ = ['TestNFWPhaseSpace']

class TestNFWPhaseSpace(TestCase):
    """ Class used to test `~halotools.empirical_models.NFWPhaseSpace`. 
    """

    def setup_class(self):
        """ Load the NFW model and build a coarse lookup table.
        """
        # self.halocat = HaloCatalog()

        self.nfw = NFWPhaseSpace()
        cmin, cmax, dc = 1, 25, 0.5
        self.nfw._setup_lookup_tables((cmin, cmax, dc))
        self.nfw.build_lookup_tables()

        Npts = 1e3
        self.c15 = np.ones(Npts) + 15
        self.c10 = np.ones(Npts) + 10
        self.c5 = np.ones(Npts) + 5

    def test_constructor(self):
        """
        """
        ### MonteCarloGalProf attributes
        assert hasattr(self.nfw, 'logradius_array')
        assert hasattr(self.nfw, 'rad_prof_func_table')
        assert hasattr(self.nfw, 'vel_prof_func_table')
        assert hasattr(self.nfw, '_mc_dimensionless_radial_distance')

        ### NFWPhaseSpace attributes
        assert hasattr(self.nfw, 'assign_phase_space')
        assert hasattr(self.nfw, 'column_keys_to_allocate')

        ### AnalyticDensityProf attributes 
        assert hasattr(self.nfw, 'circular_velocity')

        ### NFWProfile attributes
        assert hasattr(self.nfw, 'mass_density')
        assert hasattr(self.nfw, 'prof_param_keys')

        ### ConcMass
        assert hasattr(self.nfw, 'conc_NFWmodel')
        assert hasattr(self.nfw, 'conc_mass_model')

    def test_mc_unit_sphere(self):
        """ Method used to test 
        `~halotools.empirical_models.NFWPhaseSpace.mc_unit_sphere`. 
        """
        x, y, z = self.nfw.mc_unit_sphere(100, seed=43)
        pos = np.vstack([x, y, z]).T 
        norm = np.linalg.norm(pos, axis=1)
        assert np.allclose(norm, 1, rtol=1e-4)

    def test_mc_dimensionless_radial_distance(self):
        """ Method used to test 
        `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`. 
        """
        r15 = self.nfw._mc_dimensionless_radial_distance(profile_params=[self.c15], seed=43)
        r10 = self.nfw._mc_dimensionless_radial_distance(profile_params=[self.c10], seed=43)
        r5 = self.nfw._mc_dimensionless_radial_distance(profile_params=[self.c5], seed=43)

        assert np.all(r15 <= 1)
        assert np.all(r15 >= 0)
        assert np.all(r10 <= 1)
        assert np.all(r10 >= 0)
        assert np.all(r5 <= 1)
        assert np.all(r5 >= 0)

        assert np.mean(r15) < np.mean(r10) < np.mean(r5)
        assert np.median(r15) < np.median(r10) < np.median(r5)

    def test_mc_solid_sphere(self):
        """ Method used to test 
        `~halotools.empirical_models.NFWPhaseSpace.mc_solid_sphere`. 
        """
        x, y, z = self.nfw.mc_solid_sphere(profile_params=[self.c15], seed=43)
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

        t = Table({'c': self.c15})
        with pytest.raises(HalotoolsError) as exc:
            x, y, z = self.nfw.mc_solid_sphere(profile_params=[self.c15], seed=43, 
                halo_table = t)

    def test_mc_halo_centric_pos(self):
        """ Method used to test 
        `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`. 
        """
        r = 0.25
        halo_radius = np.zeros(len(self.c15)) + r
        x15, y15, z15 = self.nfw.mc_halo_centric_pos(
            halo_radius=halo_radius, profile_params=[self.c15], seed=43)
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

        x5, y5, z5 = self.nfw.mc_halo_centric_pos(
            halo_radius=halo_radius, profile_params=[self.c5], seed=43)
        pos5 = np.vstack([x5, y5, z5]).T
        norm5 = np.linalg.norm(pos5, axis=1)

        x10, y10, z10 = self.nfw.mc_halo_centric_pos(
            halo_radius=halo_radius, profile_params=[self.c10], seed=43)
        pos10 = np.vstack([x10, y10, z10]).T
        norm10 = np.linalg.norm(pos10, axis=1)

        assert np.mean(norm5) > np.mean(norm10) > np.mean(norm15)
        assert np.median(norm5) > np.median(norm10) > np.median(norm15)

        x10a, y10a, z10a = self.nfw.mc_halo_centric_pos(
            halo_radius=halo_radius*2, profile_params=[self.c10], seed=43)
        pos10a = np.vstack([x10a, y10a, z10a]).T
        norm10a = np.linalg.norm(pos10a, axis=1)

        assert np.any(norm10a > r)
        assert np.all(norm10a < 2*r)
     
        t = Table({'c': self.c15})
        with pytest.raises(HalotoolsError) as exc:
            x, y, z = self.nfw.mc_halo_centric_pos(
            halo_radius=halo_radius, profile_params=[self.c10], seed=43, halo_table = t)
        t['host_centric_distance'] = 0.
        x, y, z = self.nfw.mc_halo_centric_pos(
            halo_radius=halo_radius, profile_params=[self.c10], seed=43, halo_table = t)
        norm = t['host_centric_distance']
        assert np.all(norm > 0)
        assert np.all(norm < halo_radius)

    def test_mc_pos(self):
        """ Method used to test 
        `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`. 
        """
        r = 0.25
        halo_radius = np.zeros(len(self.c15)) + r
        x1, y1, z1 = self.nfw.mc_pos(
            halo_radius=halo_radius, profile_params=[self.c15], seed=43)
        x2, y2, z2 = self.nfw.mc_halo_centric_pos(
            halo_radius=halo_radius, profile_params=[self.c15], seed=43)
        assert np.all(x1 == x2)
        assert np.all(y1 == y2)
        assert np.all(z1 == z2)

    def test_vrad_disp_from_lookup(self):
        """ Method used to test 
        `~halotools.empirical_models.NFWPhaseSpace._vrad_disp_from_lookup`. 
        """
        x = np.random.uniform(0, 1, len(self.c15))
        vr_disp = self.nfw._vrad_disp_from_lookup(
            x = x, profile_params=[self.c15], seed=43)
    
        assert np.all(vr_disp < 1)
        assert np.all(vr_disp > 0)

    def test_mc_radial_velocity(self):
        """
        """
        npts = 1e2
        conc = 10
        carr = np.ones(npts) + conc

        mass = 1e12
        v = self.nfw.virial_velocity(mass)
        rmax = self.nfw.rmax(mass, conc)
        vmax = self.nfw.vmax(mass, conc)
        r = np.zeros(npts) + rmax
        rvir = self.nfw.halo_mass_to_halo_radius(mass)
        x = r/rvir

        v = 250.
        vvir = np.zeros_like(x) + v
        mc_vr = self.nfw.mc_radial_velocity(
            x = x, virial_velocities = vvir, profile_params = [carr])

        vr_dispersion_from_monte_carlo = np.std(mc_vr)
        assert np.allclose(vr_dispersion_from_monte_carlo, vmax, rtol=0.05)


    def test_mc_vel(self):

        npts = 1e3
        Lbox = 250
        zeros = np.zeros(npts)
        x = np.random.uniform(0, Lbox, npts)
        y = np.random.uniform(0, Lbox, npts)
        z = np.random.uniform(0, Lbox, npts)
        halo_vx = np.random.uniform(-250, 250, npts)
        halo_vy = np.random.uniform(-250, 250, npts)
        halo_vz = np.random.uniform(-250, 250, npts)
        d = np.random.uniform(0, 0.25, npts)
        rvir = np.zeros(npts) + 0.2
        conc_nfw = np.random.uniform(1.5, 15, npts)
        mass = np.zeros(npts) + 1e12
        vvir = self.nfw.virial_velocity(total_mass=mass)

        t = Table({'halo_x': x, 'halo_y': y, 'halo_z': z, 
            'host_centric_distance': d, 'halo_rvir': rvir, 'conc_NFWmodel': conc_nfw, 
            'halo_vx': halo_vx, 'halo_vy': halo_vy, 'halo_vz': halo_vz, 'halo_vvir': vvir, 
            'x': zeros, 'y': zeros, 'z': zeros, 'vx': zeros, 'vy': zeros, 'vz': zeros})

        self.nfw.mc_vel(t)








