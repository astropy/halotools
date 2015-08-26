# -*- coding: utf-8 -*-
"""
"""
import numpy as np 
from astropy import cosmology
from astropy import units as u

def density_threshold(cosmo, z, mdef):
	"""
	The threshold density for a given spherical overdensity mass definition.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	z: array_like
		Redshift. Can be a scalar or a numpy array.

	mdef: str
		String specifying the mass definition, e.g., 'vir' or '200m'. 
		
	Returns
	-----------------------------------------------------------------------------------------------
	rho: array_like
		The threshold density in physical :math:`M_{\odot}h^2/Mpc^3`. 
		Has the same dimensions as the input ``z``. 

	See also
	-----------------------------------------------------------------------------------------------
	delta_vir: The virial overdensity in units of the critical density.
	"""
	rho_crit = cosmo.critical_density(z)
	rho_crit = rho_crit.to(u.Msun/u.Mpc**3).value/cosmo.h**2

	if mdef[len(mdef) - 1] == 'c':
		delta = int(mdef[:-1])
		rho_treshold = rho_crit * delta

	elif mdef[len(mdef) - 1] == 'm':
		rho_crit0 = cosmo.critical_density0
		rho_crit0 = rho_crit0.to(u.Msun/u.Mpc**3).value/cosmo.h**2
		delta = int(mdef[:-1])
		rho_m = cosmo.Om(z)*rho_crit0
		rho_treshold = delta * rho_m

	elif mdef == 'vir':
		delta = delta_vir(cosmo, z)
		rho_treshold = rho_crit * delta

	else:
		msg = 'Invalid mass definition, %s.' % mdef
		raise Exception(msg)

	return rho_treshold

def delta_vir(cosmo, z):
	"""
	The virial overdensity in units of the critical density.
	
	This function uses the fitting formula of Bryan & Norman 1998 to determine the virial 
	overdensity. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	z: array_like
		Redshift. Can be a scalar or a numpy array.
		
	Returns
	-----------------------------------------------------------------------------------------------
	delta: array_like
		The virial overdensity.
		Has the same dimensions as the input ``z``. 

	See also
	-----------------------------------------------------------------------------------------------
	density_threshold: The threshold density for a given mass definition.
	"""
	
	x = cosmo.Om(z) - 1.0
	delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2

	return delta

def halo_mass_to_halo_radius(M, cosmo, z, mdef):
	"""
	Spherical overdensity mass from radius.
	
	This function returns a spherical overdensity halo radius for a halo mass M. Note that this 
	function is independent of the form of the density profile.

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	R: array_like
		Halo radius in physical kpc/h; has the same dimensions as M.

	See also
	-----------------------------------------------------------------------------------------------
	halo_radius_to_halo_mass: Spherical overdensity radius from mass.
	"""
	
	rho = density_threshold(cosmo, z, mdef)
	R = (M * 3.0 / 4.0 / np.pi / rho)**(1.0 / 3.0)

	return R

def halo_radius_to_halo_mass(R, cosmo, z, mdef):
	"""
	Spherical overdensity radius from mass.
	
	This function returns a spherical overdensity halo mass for a halo radius R. Note that this 
	function is independent of the form of the density profile.

	Parameters
	-----------------------------------------------------------------------------------------------
	R: array_like
		Halo radius in physical kpc/h; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	M: array_like
		Mass in :math:`M_{\odot}/h`; has the same dimensions as R.

	See also
	-----------------------------------------------------------------------------------------------
	halo_mass_to_halo_radius: Spherical overdensity mass from radius.
	"""
	
	rho = density_threshold(cosmo, z, mdef)
	M = 4.0 / 3.0 * np.pi * rho * R**3

	return M

