# -*- coding: utf-8 -*-
""" Module containing functions related to halo mass definitions, 
the relations between halo mass and radius, and the variation of these 
relations with cosmology and redshift. 

The functions contained in this module borrow heavily from the Colossus 
package developed by Benedikt Diemer, http://bdiemer.bitbucket.org. 
"""
from __future__ import (
	division, print_function, absolute_import, unicode_literals)

import numpy as np 
from astropy import cosmology
from astropy import units as u

__all__ = (['density_threshold', 'delta_vir', 
	'halo_mass_to_halo_radius', 'halo_radius_to_halo_mass'])

__author__ = ['Benedikt Diemer', 'Andrew Hearin']

def density_threshold(cosmology, redshift, mdef):
	"""
	The threshold density for a given spherical overdensity mass definition.
	
	Parameters
	--------------
	cosmology : object 
		Instance of an `~astropy.cosmology` object. 

	redshift: array_like
		Can be a scalar or a numpy array.

	mdef: str
		String specifying the halo mass definition, e.g., 'vir' or '200m'. 
		
	Returns
	---------
	rho: array_like
		The threshold density in physical :math:`M_{\odot}h^2/Mpc^3`. 
		Has the same dimensions as the input ``redshift``. 

	See also
	----------
	delta_vir: The virial overdensity in units of the critical density.
	"""
	rho_crit = cosmology.critical_density(redshift)
	rho_crit = rho_crit.to(u.Msun/u.Mpc**3).value/cosmology.h**2

	if mdef[-1] == 'c':
		delta = int(mdef[:-1])
		rho_treshold = rho_crit * delta

	elif mdef[-1] == 'm':
		rho_crit0 = cosmology.critical_density0
		rho_crit0 = rho_crit0.to(u.Msun/u.Mpc**3).value/cosmology.h**2
		delta = int(mdef[:-1])
		rho_m = cosmology.Om(redshift)*rho_crit0
		rho_treshold = delta * rho_m

	elif mdef == 'vir':
		delta = delta_vir(cosmology, redshift)
		rho_treshold = rho_crit * delta

	else:
		msg = 'Invalid mass definition, %s.' % mdef
		raise Exception(msg)

	return rho_treshold

def delta_vir(cosmology, redshift):
	"""
	The virial overdensity in units of the critical density, 
	using the fitting formula of Bryan & Norman 1998.
	
	Parameters
	--------------
	cosmology : object 
		Instance of an `~astropy.cosmology` object. 

	redshift: array_like
		Can be a scalar or a numpy array.
		
	Returns
	----------
	delta: array_like
		The virial overdensity. Has the same dimensions as the input ``redshift``. 

	See also
	-----------
	density_threshold: The threshold density for a given mass definition.
	"""
	
	x = cosmology.Om(redshift) - 1.0
	delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2

	return delta

def halo_mass_to_halo_radius(mass, cosmology, redshift, mdef):
	"""
	Spherical overdensity radius as a function of the input mass. 

	Note that this function is independent of the form of the density profile.

	Parameters
	------------
	mass: array_like
		Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

	cosmology : object 
		Instance of an `~astropy.cosmology` object. 

	redshift: array_like
		Can either be a scalar, or a numpy array of the same dimension as the input ``mass``. 

	mdef: str
		String specifying the halo mass definition, e.g., 'vir' or '200m'. 
		
	Returns
	--------
	radius: array_like
		Halo radius in physical Mpc/h; has the same dimensions as input ``mass``.

	See also
	---------------
	halo_radius_to_halo_mass: Spherical overdensity radius from mass.
	"""
	
	rho = density_threshold(cosmology, redshift, mdef)
	radius = (mass * 3.0 / 4.0 / np.pi / rho)**(1.0 / 3.0)

	return radius

def halo_radius_to_halo_mass(radius, cosmology, redshift, mdef):
	"""
	Spherical overdensity mass as a function of the input radius. 

	Note that this function is independent of the form of the density profile.

	Parameters
	------------
	radius: array_like
		Halo radius in physical Mpc/h; can be a scalar or a numpy array.

	cosmology : object 
		Instance of an `~astropy.cosmology` object. 

	redshift: array_like
		Can either be a scalar, or a numpy array of the same dimension as the input ``radius``. 

	mdef: str
		String specifying the halo mass definition, e.g., 'vir' or '200m'. 
		
	Returns
	---------
	mass: array_like
		Total halo mass in :math:`M_{\odot}/h`; has the same dimensions as the input ``radius``. 

	"""
	
	rho = density_threshold(cosmology, redshift, mdef)
	mass = 4.0 / 3.0 * np.pi * rho * radius**3
	return mass

