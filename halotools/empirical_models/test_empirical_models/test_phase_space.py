#!/usr/bin/env python

from unittest import TestCase

import numpy as np 
from astropy.table import Table 
from ...sim_manager import HaloCatalog
from ..phase_space_models import NFWPhaseSpace

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
        r15 = self.nfw._mc_dimensionless_radial_distance(self.c15, seed=43)
        r10 = self.nfw._mc_dimensionless_radial_distance(self.c10, seed=43)
        r5 = self.nfw._mc_dimensionless_radial_distance(self.c5, seed=43)

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
        `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`. 
        """
        x, y, z = self.nfw.mc_solid_sphere(self.c15, seed=43)
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









