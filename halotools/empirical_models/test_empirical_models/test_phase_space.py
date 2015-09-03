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
        cmin, cmax, dc = 1, 15, 25
        self.nfw._setup_lookup_tables((cmin, cmax, dc))
        self.nfw.build_lookup_tables()

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
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_unit_sphere`. 
        """
        x, y, z = self.nfw.mc_unit_sphere(100)
        pos = np.vstack([x, y, z]).T 
        norm = np.linalg.norm(pos, axis=1)
        assert np.allclose(norm, 1, rtol=1e-4)










