# -*- coding: utf-8 -*-
"""

This module contains the classes related to 
the radial profiles of dark matter halos.

"""

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

from utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 
import defaults

import astropy.cosmology as cosmology
from astropy import units as u



##################################################################################


@six.add_metaclass(ABCMeta)
class HaloProfileModel(object):
    """ Container class for any halo profile model. """

    def __init__(self):
    	pass

    @abstractmethod
    def density_profile(self, r, *args):
    	raise NotImplementedError("All halo profile models must include a mass_density method")

    @abstractmethod
    def cumulative_mass_PDF(self, r, *args):
    	raise NotImplementedError("All halo profile models must include a cumulative_mass_PDF method")


class NFWProfile(HaloProfileModel):

	def __init__(self, delta_vir=360.0, cosmology=cosmology.WMAP5):

		self.delta_vir = delta_vir
		self.cosmology = cosmology
		littleh = self.cosmology.H0/100.0
		crit_density = (
			self.cosmology.critical_density(0).to(u.Msun/u.Mpc**3)/littleh**2)
		self.cosmic_matter_density = crit_density*self.cosmology.Om0

	def _g(self, x):
		denominator = np.log(1.0+x) - (x/(1.0+x))
		return 1./denominator

	def rho_s(self, c):
		return (self.delta_vir/3.)*c*c*c*self._g(c)*self.cosmic_matter_density

	def density_profile(self, r, c):
		numerator = self.rho_s(c)
		denominator = (c*r)*(1.0 + c*r)*(1.0 + c*r)
		return numerator / denominator

	def cumulative_mass_PDF(self, r, c):
		return self._g(c) / self._g(r*c)




















