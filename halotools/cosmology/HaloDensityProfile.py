###################################################################################################
#
# Concentration.py 		(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# This module implements soem convenient functions related to halo density profiles. Functions
# include:
#
# pseudoEvolve           Assume a static NFW profile, and compute how spherical overdensity mass 
#                        and radius evolve with redshift due to the evolving reference density (an 
#                        effect called pseudo-evolution).
#
# convertMassDefinition  Convert one spherical overdensity mass definition to another.
#
# Other functions include threshold density, Delta(z), NFW parameters / enclosed mass etc, as 
# well as finding a spherical overdensity radius and mass for arbitrary mass profiles. Mass 
# definitions are given in a string format. Valid inputs are 
#
# vir           A varying overdensity mass definition, implemented using the fitting formula of 
#               Bryan & Norman 1998.
# ****c         An integer number times the critical density of the universe, such as 200c. 
# ****m         An integer number times the matter density of the universe, such as 200m. 
#
# NOTE:         For virtually all functions in this module, a cosmology must be set. See the 
#               documentation in Cosmology.py.
#
###################################################################################################

import math
import numpy
import scipy.interpolate
import scipy.optimize

import Cosmology

###################################################################################################
# PSEUDO-EVOLUTION
###################################################################################################

# This function computes the evolution of spherical overdensity mass and radius due to a changing 
# reference density, an effect called 'pseudo-evolution'. See 
#
#     Diemer, More & Kravtsov, 2013 ApJ 766, 25
#
# for a detailed description. Here, we assume that the density profile is an NFW profile and that 
# it does not change over redshift. In order to describe this static profile, the user passes mass
# in Msun/h and concentration at a redshift z_i and for a mass definition mdef_i. The function 
# computes M and R at another redshift z_f and for a mass definition z_f. If the mass definitions
# are the same, the difference between the input and output masses is the pseudo-evolution of this
# mass definition between z_i and z_f. 
# 
# Another special case is z_i = z_f, in which case the function converts one mass definition to 
# another. Use the convertMassDefinition() function below for this case. See the documentation at 
# the top of this file for valid mass definition formats.

def pseudoEvolve(M_i, c_i, z_i, mdef_i, z_f, mdef_f):

	# This equation is 0 when the enclosed density matches the given rho_t.
	def MvirRootEqn(x, rhos, rhot):
		return rhos * NFW_mu(x) * 3.0 / x**3 - rhot

	# Get profile parameters at initial redshift z_i
	_, rs, rhos = NFW_getParameters(M_i, c_i, z_i, mdef_i)

	# Get threshold density at target redshift z_f
	rhot = rhoThreshold(z_f, mdef_f)
	args = rhos, rhot
	x = scipy.optimize.brentq(MvirRootEqn, c_i / 100.0, c_i * 100.0, args)

	R = x * rs
	M = 4.0 * math.pi / 3.0 * R**3 * rhot

	return M, R

###################################################################################################

# Get the spherical overdensity mass, radius and concentration for a mass definition mdef_out, 
# given the mass and concentration at redshift z and for mass definition mdef_in. See the 
# documentation at the top of this file for valid mass definition formats.

def convertMassDefinition(M, c, z, mdef_in, mdef_out):

	Mnew, Rnew = pseudoEvolve(M, c, z, mdef_in, z, mdef_out)
	_, rs, _ = NFW_getParameters(M, c, z, mdef_in)
	cnew = Rnew / rs
	
	return Mnew, Rnew, cnew

###################################################################################################
# FUNCTIONS RELATED TO GENERAL DENSITY PROFILES
###################################################################################################

# The virial overdensity, using the fitting formula of Bryan & Norman 1998.

def deltaVir(z):

	cosmo = Cosmology.getCurrent()
	x = cosmo.Om(z) - 1.0
	Delta = 18 * math.pi**2 + 82.0 * x - 39.0 * x**2

	return Delta

###################################################################################################
	
# Returns a density threshold in Msun h^2 / kpc^3 at a redshift z. See the documentation at the top
# of this file for valid mass definition formats.

def rhoThreshold(z, mdef):

	cosmo = Cosmology.getCurrent()
	rho_crit = Cosmology.AST_rho_crit_0_kpc3 * cosmo.Ez(z)**2

	if mdef[len(mdef) - 1] == 'c':
		delta = int(mdef[:-1])
		rho_treshold = rho_crit * delta

	elif mdef[len(mdef) - 1] == 'm':
		delta = int(mdef[:-1])
		rho_m = Cosmology.AST_rho_crit_0_kpc3 * cosmo.Om0 * (1.0 + z)**3
		rho_treshold = delta * rho_m

	elif mdef == 'vir':
		delta = deltaVir(z)
		rho_treshold = rho_crit * delta

	else:
		raise Exception(("Invalid mass definition" + mdef))

	return rho_treshold

###################################################################################################

# This function returns a spherical overdensity halo radius in kpc / h for a halo mass M in Msun/h,
# a redshift z, and a mass definition (see the documentation at the top of this file for valid mass 
# definition formats.)

def R_Delta(M, z, mdef):

	rho = rhoThreshold(z, mdef)
	r = (M * 3.0 / 4.0 / math.pi / rho)**(1.0 / 3.0)

	return r

###################################################################################################

# This function returns the spherical overdensity mass (in Msun/h) and radius (in kpc/h) given a 
# mass definition, redshift, and halo enclosed mass profile. The profile must be a 2D array with
# radii in kpc/h in the first column and enclosed mass in Msun/h in the second column. See the 
# documentation at the top of this file for valid mass definition formats.

def getMR(M_enclosed, z, mdef):

	# This equation is 0 when the enclosed density matches the given rho_t.
	def MvirRootEqn(r, spline, rhot):
		return spline(r) / 4.0 / math.pi * 3.0 / r**3 - rhot

	rhot = rhoThreshold(z, mdef)
	N = len(M_enclosed[0])
	spline = scipy.interpolate.InterpolatedUnivariateSpline(M_enclosed[0], M_enclosed[1])

	args = spline, rhot
	R = scipy.optimize.brentq(MvirRootEqn, M_enclosed[0][0], M_enclosed[0][N - 1], args)
	M = 4.0 * math.pi / 3.0 * R**3 * rhot

	return M, R

###################################################################################################
# FUNCTIONS RELATED TO NFW PROFILES
###################################################################################################

# The mu(c) function that appears in the expression for the enclosed mass in NFW profiles.

def NFW_mu(c):

	return numpy.log(1.0 + c) - c / (1.0 + c)

###################################################################################################

# The enclosed mass for an NFW profile with central density rhos (in Msun h^2 / kpc^3) and sclae 
# radius rs (in kpc/h), as a function of x = r / rs. Note that there is no / 3 factor in the mass 
# of the NFW profile.

def NFW_Mr(rhos, rs, x):

	return 4.0 * math.pi * rs**3 * rhos * NFW_mu(x)

###################################################################################################

# Convert mass and concentration into the native NFW parameters: central density (in Msun h^2 / 
# kpc^3) and scale radius (in kpc/h). In addition, the spherical overdensity radius R (in kpc/h) is 
# returned as well.

def NFW_getParameters(M, c, z, mdef):

	R = R_Delta(M, z, mdef)
	rs = R / c
	rhos = M / rs**3 / 4.0 / math.pi / NFW_mu(c)

	return R, rs, rhos

###################################################################################################
