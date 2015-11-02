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
	"""
	"""

	def setup_class(self):
		pass


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
					assert np.allclose(r1, r0, rtol = 1e-3)

	def test_delta_vir(self):
		bn98_result = delta_vir(WMAP9, 5.0)
		assert np.allclose(bn98_result, 18.*np.pi**2, rtol=0.01)


	def test_density_threshold(self):

		z = 5.0
		rho_crit = WMAP9.critical_density(z)
		rho_crit = rho_crit.to(u.Msun/u.Mpc**3).value/WMAP9.h**2
		rho_m = WMAP9.Om(z)*rho_crit

		wmap9_200c_z5 = density_threshold(WMAP9, z, '200c')/rho_crit
		assert np.allclose(wmap9_200c_z5, 200.0, rtol=0.01)

		wmap9_2500c_z5 = density_threshold(WMAP9, z, '2500c')/rho_crit
		assert np.allclose(wmap9_2500c_z5, 2500.0, rtol=0.01)

		wmap9_200m_z5 = density_threshold(WMAP9, z, '200m')/rho_m
		assert np.allclose(wmap9_200m_z5, 200.0, rtol=0.01)
		

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





