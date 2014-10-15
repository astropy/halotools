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
# convertMassDefinition  Convert one spherical overdensity mass definition to another, assuming
#                        an NFW profile.
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
import scipy.integrate

import Utilities
import Cosmology
import Concentration

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
#
# The function needs to assume a form of the density profile as a function of M and c. By default,
# the NFW profile is used, but the user can also choose profile = 'dk14' for the DK14 profile. Note
# that the latter version is much slower.

def convertMassDefinition(M, c, z, mdef_in, mdef_out, profile = 'nfw'):

	if profile == 'nfw':
		
		Mnew, Rnew = pseudoEvolve(M, c, z, mdef_in, z, mdef_out)
		_, rs, _ = NFW_getParameters(M, c, z, mdef_in)
		cnew = Rnew / rs
	
	elif profile == 'dk14':
		
		par = DK14_getParameters(M, mdef_in, z, c = c, selected = 'by_mass', part = 'inner')
		
		# The DK14 profile "lives" in R200m space. Thus, R200m is automatically computed.
		if mdef_out == '200m':
			Rnew = par.R200m
		else:
			Rnew = DK14_getR(z, mdef_out, par)
			
		Mnew = M_Delta(Rnew, z, mdef_out)
		cnew = Rnew / par.rs
		
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
	R = (M * 3.0 / 4.0 / math.pi / rho)**(1.0 / 3.0)

	return R

###################################################################################################

# This function returns a spherical overdensity halo mass in Msun / h for a halo radius R in kpc/h,
# a redshift z, and a mass definition (see the documentation at the top of this file for valid mass 
# definition formats.)

def M_Delta(R, z, mdef):

	rho = rhoThreshold(z, mdef)
	M = 4.0 / 3.0 * math.pi * rho * R**3

	return M

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

# Convert mass and concentration into the native NFW parameters: central density (in Msun h^2 / 
# kpc^3) and scale radius (in kpc/h). In addition, the spherical overdensity radius R (in kpc/h) is 
# returned as well.

def NFW_getParameters(M, c, z, mdef):

	R = R_Delta(M, z, mdef)
	rs = R / c
	rhos = M / rs**3 / 4.0 / math.pi / NFW_mu(c)

	return R, rs, rhos

###################################################################################################

# The density for an NFW profile with central density rhos (in Msun h^2 / kpc^3) and scale radius 
# rs (in kpc / h), as a function of x = r / rs.

def NFW_rho(rhos, x):

	return rhos / x / (1.0 + x)**2

###################################################################################################

# The enclosed mass for an NFW profile with central density rhos (in Msun h^2 / kpc^3) and scale 
# radius rs (in kpc / h), as a function of x = r / rs.

def NFW_M(rhos, rs, x):

	return 4.0 * math.pi * rs**3 * rhos * NFW_mu(x)

###################################################################################################

# The surface density (in units of Msun h / kpc^2)  with central density rhos (in Msun h^2 / kpc^3)
# of an NFW profile and scale radius rs (in kpc / h), as a function of x = r / rs.

def NFW_Sigma(rhos, rs, x):

	if not Utilities.isArray(x):
		x_use = numpy.array([x])
	else:
		x_use = x
	
	Sigma = 0.0 * x_use
	for i in range(len(x_use)):
		
		xx = x_use[i]
		xx2 = xx**2
		
		if abs(xx - 1.0) < 1E-2:
			fx = 0.0
		else:
			if xx > 1.0:
				fx = 1.0 - 2.0 / math.sqrt(xx2 - 1.0) * math.atan(math.sqrt((xx - 1.0) / (xx + 1.0)))
			else:
				fx = 1.0 - 2.0 / math.sqrt(1.0 - xx2) * math.atanh(math.sqrt((1.0 - xx) / (1.0 + xx)))
	
		Sigma[i] = 2.0 * rhos * rs / (x_use[i]**2 - 1.0) * fx

	if not Utilities.isArray(x):
		Sigma = Sigma[0]

	return Sigma

###################################################################################################
# DK14 PROFILE - LOW-LEVEL FUNCTIONS
###################################################################################################

# This object specifies the parameters of the Diemer & Kravtsov 2014 halo density profile. The 
# full parameter freedom should almost never be used, see the simplified functions below. Here we 
# use a power-law outer profile.
#
# The profile has two parts, the 1-halo term ('inner') and the 2-halo term ('outer'). By default, 
# the function returns their sum ('both'). The other parameters have the following meaning:
#
# R200m		The radius that encloses and average overdensity of 200 rho_m(z). This parameter is of
#           fundamental importance in the DK14 profile, as the outer profile is most universal in
#           units of R200m.
# rho_s		The central scale density, in units of Msun h^2 / kpc^3
# rho_m     The mean matter density of the universe, in units of Msun h^2 / kpc^3
# rs        The scale radius in units of kpc / h
# rt        The radius where the profile steepens, in units of kpc / h
# alpha     Determines how quickly the slope of the inner Einasto profile steepens
# beta      Sharpness of the steepening
# gamma	    Asymptotic negative slope of the steepening term
# be        Normalization of the power-law outer profile
# se        Slope of the power-law outer profile

class DK14Parameters():
	
	def __init__(self):
		
		self.R200m = 0.0
		self.rho_s = 0.0
		self.rho_m = 0.0
		self.rs = 0.0
		self.rt = 0.0
		self.alpha = 0.0
		self.beta = 0.0
		self.gamma = 0.0
		self.be = 0.0
		self.se = 0.0
		
		self.part = 'both'
		
		return

###################################################################################################

# Get the parameter values for the DK14 profile that should be fixed, or can be determined from the 
# peak height or mass accretion rate. If selected is 'by_mass', only nu must be passed. If selected 
# is 'by_accretion_rate', then both z and Gamma must be passed.

def DK14_getFixedParameters(selected, nu = None, z = None, Gamma = None):

	cosmo = Cosmology.getCurrent()

	if selected == 'by_mass':
		beta = 4.0
		gamma = 8.0
		rt_R200m = abs(1.9 - 0.18 * nu)
	elif selected == 'by_accretion_rate':
		beta = 6.0
		gamma = 4.0
		rt_R200m = (0.425 + 0.402 * cosmo.Om(z)) * (1 + 2.148 * numpy.exp(-Gamma / 1.962))
	else:
		msg = "HaloDensityProfile.DK14_getFixedParameters: Unknown sample selection, %s." % (selected)
		raise Exception(msg)
	
	alpha = 0.155 + 0.0095 * nu**2
			
	return alpha, beta, gamma, rt_R200m

###################################################################################################

# Get the DK14 parameters that correspond to a profile with a particular mass MDelta in some mass
# definition mdef. Optinally, the user can determine the concentration c - otherwise, 
#
# MDelta  		Mass
# mdef			The corresponding mass definition (see top of the file for valid formats)
# z				Redshift
# c				(Optional) concentration. If this parameter is not passed, c is estimated using 
#               the model of Diemer & Kravtsov 2014b. 
# selected		Is the sample selected 'by_mass' or 'by_accretion_rate'? This changes some of the 
#               fixed parameters.
# Gamma			The mass accretion rate as defined in DK14. This parameter only needs to be passed 
#               if selected is 'by_accretion_rate'.
# part          Can be 'both' or 'inner'. This parameter is simply passed into the return
#               structure. The value 'outer' makes no sense in this function, since the outer
#               profile alone cannot be normalized to have the mass MDelta.
# be, se        Parameters for the outer profile. These only need to be passed if part = 'both' or 
#               part = 'outer'.
# acc_warn		If the function achieves a relative accuracy in matching MDelta less than this value,
#               a warning is printed.
# acc_err		If the function achieves a relative accuracy in matching MDelta less than this value,
#               an exception is raised.

def DK14_getParameters(MDelta, mdef, z, c = None, selected = 'by_mass', Gamma = None, \
				part = 'both', be = None, se = None, acc_warn = 0.01, acc_err = 0.05):
	
	# Declare shared variables; these parameters are advanced during the iterations
	par2 = {}
	par2['Rvir'] = 0.0
	par2['RDelta'] = 0.0
	par2['nu'] = 0.0
	
	par = DK14Parameters()
	
	RTOL = 0.01
	MTOL = 0.01
	GUESS_TOL = 2.5
		
	def radius_diff(R200m, par, par2, Gamma, rho_target, rho_vir, R_target):
		
		# Remember the parts we need to evaluate; this will get overwritten
		part_true = par.part
		par.R200m = R200m
		
		# Set nu_vir from previous Rvir
		Mvir = M_Delta(par2['Rvir'], z, 'vir')
		par2['nu'] = cosmo.M_to_nu(Mvir, z)

		# Set profile parameters
		par.alpha, par.beta, par.gamma, rt_R200m = DK14_getFixedParameters(selected, \
												nu = par2['nu'], z = z, Gamma = Gamma)
		par.rt = rt_R200m * R200m

		# Find rho_s; this can be done without iterating
		par.rho_s = 1.0
		par.part = 'inner'
		M200m = M_Delta(R200m, z, '200m')
		Mr_inner = DK14_M(R200m, par, relative_acc = MTOL)
		
		if part == 'both':
			par.part = 'outer'
			Mr_outer = DK14_M(R200m, par, relative_acc = MTOL)
		elif part == 'inner':
			Mr_outer = 0.0
		else:
			msg = "HaloDensityProfile.DK14_getParameters: Invalid value for part, %s." % (part)
			raise Exception(msg)
			
		par.rho_s = (M200m - Mr_outer) / Mr_inner
		par.part = part_true

		# Now compute MDelta and Mvir from this new profile
		par2['RDelta'] = DK14_getR_lowlevel(par2['RDelta'], rho_target, par, \
							mass_acc = MTOL, radius_acc = RTOL, guess_tolerance = GUESS_TOL)
		par2['Rvir'] = DK14_getR_lowlevel(par2['Rvir'], rho_vir, par, \
							mass_acc = MTOL, radius_acc = RTOL, guess_tolerance = GUESS_TOL)
		
		return par2['RDelta'] - R_target
	
	# The user needs to set a cosmology before this function can be called
	cosmo = Cosmology.getCurrent()
	
	# Get concentration if the user hasn't supplied it, compute scale radius
	if c == None:
		c = Concentration.concentration(MDelta, z, statistic = 'median', mdef = mdef)
	R_target = R_Delta(MDelta, z, mdef)
	par2['RDelta'] = R_target
	par.rs = R_target / c
	
	# Take a guess at nu_vir and R200m
	if mdef == 'vir':
		Mvir = MDelta
	else:
		Mvir, par2['Rvir'], _ = convertMassDefinition(MDelta, c, z, mdef, 'vir')
	par2['nu'] = cosmo.M_to_nu(Mvir, z)
	
	if mdef == '200m':
		R200m_guess = R_Delta(MDelta, z, '200m')
	else:
		_, R200m_guess, _ = convertMassDefinition(MDelta, c, z, mdef, '200m')
	
	# Iterate to find an M200m for which the desired mass is correct
	par.rho_m = cosmo.matterDensity(z)
	par.be = be
	par.se = se
	par.part = part
	rho_target = rhoThreshold(z, mdef)
	rho_vir = rhoThreshold(z, 'vir')
	args = par, par2, Gamma, rho_target, rho_vir, R_target
	par.R200m = scipy.optimize.brentq(radius_diff, R200m_guess / 1.3, R200m_guess * 1.3, \
						args = args, xtol = RTOL)

	# Check the accuracy of the result; M should be very close to MDelta now
	M_result = M_Delta(par2['RDelta'], z, mdef)
	err = (M_result - MDelta) / MDelta
	
	if abs(err) > acc_warn:
		print(('WARNING: DK14_getParameters converged to an accuracy of %.1f percent.' % (abs(err) * 100.0)))
	if abs(err) > acc_err:
		msg = 'DK14_getParameters converged to an accuracy of %.1f percent.' % (abs(err) * 100.0)
		raise Exception(msg)
	
	return par

###################################################################################################

# This function returns the spherical overdensity mass (in Msun / h) and radius (in kpc / h) given 
# a mass definition, redshift, the parameters of the DK14 profile (which can be computed with the 
# getParameters function above).

def DK14_getR(z, mdef, par):

	# We know R200m and thus M200m; from those parameters we can guess what R would be for an 
	# NFW profile and use this as an initial guess.
	M200m = M_Delta(par.R200m, z, mdef)
	rho_threshold = rhoThreshold(z, mdef)
	_, R_guess, _ = convertMassDefinition(M200m, par.R200m / par.rs, z, '200m', mdef)
	R = DK14_getR_lowlevel(R_guess, rho_threshold, par)

	return R

###################################################################################################

# Low-level function to compute a spherical overdensity radius given the parameters of a DK14 
# profile, the desired overdensity threshold, and an initial guess. A more user-friendly version
# can be found above (DK14_getMR).

def DK14_getR_lowlevel(R_guess, rho_threshold, par, \
			mass_acc = 1E-4, radius_acc = 1E-4, guess_tolerance = 5.0):

	def overdensity(R, rho_threshold, par):
		rho = 3.0 * DK14_M(R, par, relative_acc = mass_acc) / (4.0 * numpy.pi * R**3)
		return rho - rho_threshold
	
	args = rho_threshold, par
	R = scipy.optimize.brentq(overdensity, R_guess / guess_tolerance, R_guess * guess_tolerance, \
							args = args, xtol = radius_acc)
	
	return R

###################################################################################################

# The density of the DK14 profile as a function of radius (in kpc / h) and the profile parameters.

def DK14_rho(r, par):
	
	rho = 0.0 * r
	
	if par.part in ['inner', 'both']:
		inner = par.rho_s * numpy.exp(-2.0 / par.alpha * ((r / par.rs) ** par.alpha - 1.0))
		fT = (1.0 + (r / par.rt) ** par.beta) ** (-par.gamma / par.beta)
		rho += inner * fT
	
	if par.part in ['outer', 'both']:
		outer = par.rho_m * (par.be * (r / 5.0 / par.R200m) ** (-par.se) + 1.0)
		rho += outer
	
	return rho

###################################################################################################

# The logarithmic slope of the density of the DK14 profile as a function of radius (in kpc / h) 
# and the profile parameters.

def DK14_rho_der(r, par):
	
	rho = 0.0 * r
	drho_dr = 0.0 * r
	
	if par.part in ['inner', 'both']:
		inner = par.rho_s * numpy.exp(-2.0 / par.alpha * ((r / par.rs) ** par.alpha - 1.0))
		d_inner = inner * (-2.0 / par.rs) * (r / par.rs)**(par.alpha - 1.0)	
		fT = (1.0 + (r / par.rt) ** par.beta) ** (-par.gamma / par.beta)
		d_fT = (-par.gamma / par.beta) * (1.0 + (r / par.rt) ** par.beta) ** (-par.gamma / par.beta - 1.0) * \
			par.beta / par.rt * (r / par.rt) ** (par.beta - 1.0)
		rho += inner * fT
		drho_dr += inner * d_fT + d_inner * fT

	if par.part in ['outer', 'both']:
		outer = par.rho_m * (par.be * (r / 5.0 / par.R200m) ** (-par.se) + 1.0)
		d_outer = par.rho_m * par.be * (-par.se) / 5.0 / par.R200m * (r / 5.0 / par.R200m) ** (-par.se - 1.0)
		rho += outer
		drho_dr += d_outer
	
	der = drho_dr * r / rho
	
	return der

###################################################################################################

# The mass of the DK14 profile inside a radius (in kpc / h).

def DK14_M(r, par, relative_acc = 1E-5):
	
	def M_integral(r, par):
		return 4.0 * numpy.pi * r**2 * DK14_rho(r, par)
	
	if not Utilities.isArray(r):
		r_use = numpy.array([r])
	else:
		r_use = r
	
	Mr = 0.0 * r_use
	for i in range(len(r_use)):	
		Mr[i], _ = scipy.integrate.quad(M_integral, 0.0, r_use[i], args = par, \
								full_output = False, epsrel = relative_acc)

	if not Utilities.isArray(r):
		Mr = Mr[0]

	return Mr

###################################################################################################

# The projected (surface) density of the DK14 profile as a function of radius (in kpc / h) and 
# the profile parameters.

def DK14_Sigma(r, par, relative_acc = 1E-5, max_interations = 100):
	
	def Sigma_integral(r, R, par):
		return 2.0 * r * DK14_rho(r, par) / numpy.sqrt(r**2 - R**2)

	if not Utilities.isArray(r):
		r_use = numpy.array([r])
	else:
		r_use = r
	
	Sigma = 0.0 * r_use
	for i in range(len(r_use)):
		args = r_use[i], par
		Sigma[i], _ = scipy.integrate.quad(Sigma_integral, r_use[i], numpy.inf, args = args, \
										epsrel = relative_acc, limit = max_interations)

	if not Utilities.isArray(r):
		Sigma = Sigma[0]

	return Sigma

###################################################################################################
