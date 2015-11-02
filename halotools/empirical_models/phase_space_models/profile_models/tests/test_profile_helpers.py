#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

import numpy as np 

from unittest import TestCase
import pytest 

from astropy.cosmology import WMAP9, Planck13
from astropy import units as u

from ..profile_helpers import *
from .....custom_exceptions import HalotoolsError


class TestProfileHelpers(TestCase):

	def setup_class(self):
		rho_crit = WMAP9.critical_density(0.)
		self.rho_crit_wmap9 = rho_crit.to(u.Msun/u.Mpc**3).value/WMAP9.h**2

		rho_crit = Planck13.critical_density(0.)
		self.rho_crit_planck13 = rho_crit.to(u.Msun/u.Mpc**3).value/Planck13.h**2


	def test_halo_radius_to_halo_mass(self):
		""" Check that the radius <==> mass functions are 
		proper inverses of one another for a range of mdef, cosmology, and redshift
		"""
		r0 = 0.25

		for cosmology in (WMAP9, Planck13):
			for redshift in (0, 1, 5, 10):
				for mdef in ('vir', '200m', '2500c'):
					m1 = halo_radius_to_halo_mass(r0, cosmology, redshift, mdef)
					r1 = halo_mass_to_halo_radius(m1, cosmology, redshift, mdef)
					assert r1 == r0

	def test_delta_vir(self):
		bn98_result = delta_vir(WMAP9, 0.0)
		# assert np.allclose(bn98_result, 198, rtol=0.1)


	def test_density_threshold(self):
		result10_wmap9 = density_threshold(WMAP9, 3., 'vir')
		result10_wmap9b = delta_vir(WMAP9, 3.)

		x = result10_wmap9/self.rho_crit_wmap9
		# assert np.allclose(x, result10_wmap9b, rtol=0.1)

	def test_density_threshold_error_handling(self):

		with pytest.raises(HalotoolsError):
			result = density_threshold(WMAP9, 0.0, 'Jose Canseco')

		with pytest.raises(HalotoolsError):
			result = density_threshold(WMAP9, 0.0, '250.m')

		with pytest.raises(HalotoolsError):
			result = density_threshold(WMAP9, 0.0, '250b')

		with pytest.raises(HalotoolsError):
			result = density_threshold(WMAP9, 0.0, '-250m')

		with pytest.raises(HalotoolsError):
			result = density_threshold('Jose Canseco', 0.0, 'vir')





