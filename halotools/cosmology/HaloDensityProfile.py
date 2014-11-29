###################################################################################################
#
# Concentration.py 		(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# This module implements some fitting functions for the density profile, as well as some functions
# that are independent of the form of the profile. The former are implemented as children of the 
# HaloDensityProfile base class. This class represents a density profile in physical units (i.e., 
# independent of redshift). The following functions MUST be overloaded by a child class:
#
# __init__		         The constructor creates the profile class, given some description of the 
#                        density profile that depends on its form.
# density                The density as a function of radius.
# 
# The following functions CAN be overloaded by child classes.  Note that many of the generic 
# implementations use numerical differentiation and integration. If the exact solutions can be 
# expressed analytically, these functions should be overloaded for better performance:
#
# densityDerivativeLin   The linear derivative of density, d rho / d r.
# densityDerivativeLog   The logarithmic derivative of density, d log(rho) / d log(r).
# enclosedMass           The enclosed mass within radius r.
# surfaceDensity         The projected surface density at radius r.
# thresholdEquation      A helper function that is zero when the enclosed density meets a threshold.
# RDelta                 The spherical overdensity radius for a given mass definition and redshift.
# RMDelta                The spherical overdensity radius and mass for a given mass definition and 
#                        redshift.
# MDelta                 The spherical overdensity mass for a given mass definition and redshift.
#
# -------------------------------------------------------------------------------------------------
# 
# Currently implemented profiles include:
#
# NFWProfile             The Navarro-Frenk-White profile of Navarro et al. 1997.
# DK14Profile            The profile of Diemer & Kravtsov 2014.
# SplineDensityProfile   A user-defined density profile, where either density, mass, or both are 
#                        interpolated using splines.
#
# -------------------------------------------------------------------------------------------------
#
# A few functions do depend on the density profile, but are abstracted for convenience. Here, the 
# user can choose which form of the density profile is used for the operation.
#
# pseudoEvolve           Assume a static density profile, and compute how spherical overdensity mass 
#                        and radius evolve with redshift due to the evolving reference density (an 
#                        effect called pseudo-evolution).
#
# changeMassDefinition   Convert one spherical overdensity mass definition to another, assuming
#                        a particular density profile.
#
# -------------------------------------------------------------------------------------------------
#
# Functions that are not specific to a particular form of the density profile are the following:
#
# M_to_R                 Convert a spherical overdensity mass into a radius, given a redshift and
#                        mass definition.
# R_to_M                 Convert a spherical overdensity radius into a mass, given a redshift and
#                        mass definition.
# deltaVir               Compute the Bryan & Norman 1998 approximation of the virial overdensity.
# densityThreshold       Compute a density threshold in physical units given a mass definition and 
#                        redshift.
# haloBias               Compute the halo bias using the approximation of Tinker et al. 2010.

# -------------------------------------------------------------------------------------------------
#
# Mass definitions are given in a string format. Valid inputs are:
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
import scipy.misc
import scipy.optimize
import scipy.integrate
import scipy.interpolate

import Utilities
import Cosmology
import HaloConcentration

###################################################################################################
# ABSTRACT BASE CLASS FOR HALO DENSITY PROFILES
###################################################################################################

# This class represents a density profile in physical units (i.e., independent of redshift). See 
# the documentation at the top of this file for details.

class HaloDensityProfile():

	def __init__(self):
		
		# The radial limits within which the profile is valid
		self.rmin = 0.0
		self.rmax = numpy.inf
		
		# The radial limits within which we search for spherical overdensity radii. These limits 
		# can be set much tighter for better performance.
		self.min_RDelta = 0.001
		self.max_RDelta = 10000.0
		
		return
	
	###############################################################################################

	# Density as a function of radius; besides the constructor, this is the only function that 
	# MUST be overwritten by child classes.
	
	def density(self, r):
		
		msg = 'The density(r) function must be overwritten by child classes.'
		raise Exception(msg)
	
		return

	###############################################################################################

	# The linear derivative of density, evaluated numerically.
	
	def densityDerivativeLin(self, r):
		
		r_use, is_array = Utilities.getArray(r)
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(self.density, r_use[i], dx = 0.001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]
		
		return density_der

	###############################################################################################

	# The logarithmic derivative of density, d log(rho) / d log(r), evaluated numerically.
	
	def densityDerivativeLog(self, r):
		
		def logRho(logr):
			return numpy.log(self.density(numpy.exp(logr)))

		r_use, is_array = Utilities.getArray(r)
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(logRho, numpy.log(r_use[i]), dx = 0.0001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]

		return density_der
		
	###############################################################################################

	# The mass (in Msun / h) enclosed within a radius r (in kpc / h), integrated numerically.
	
	def enclosedMass(self, r, accuracy = 1E-6):
		
		def integrand(r):
			return self.density(r) * 4.0 * numpy.pi * r**2

		r_use, is_array = Utilities.getArray(r)
		M = 0.0 * r_use
		for i in range(len(r_use)):	
			M[i], _ = scipy.integrate.quad(integrand, self.rmin, r_use[i], epsrel = accuracy)
		if not is_array:
			M = M[0]
	
		return M

	###############################################################################################

	# The projected surface density (in Msun h / kpc^2) as a function of radius (in kpc / h).
	
	def surfaceDensity(self, r, accuracy = 1E-6):
		
		def integrand(r, R):
			ret = 2.0 * r * self.density(r) / numpy.sqrt(r**2 - R**2)
			return ret

		r_use, is_array = Utilities.getArray(r)
		surfaceDensity = 0.0 * r_use
		for i in range(len(r_use)):	
			
			if r_use[i] >= self.rmax:
				msg = 'Cannot compute surface density for radius %.2e since rmax is %.2e.' % (r_use[i], self.rmax)
				raise Exception(msg)
			
			surfaceDensity[i], _ = scipy.integrate.quad(integrand, r_use[i], self.rmax, args = r_use[i], \
											epsrel = accuracy, limit = 1000)
		if not is_array:
			surfaceDensity = surfaceDensity[0]

		return surfaceDensity
	
	###############################################################################################

	# This equation is 0 when the enclosed density matches the given density_threshold, and is used 
	# when numerically determining spherical overdensity radii.
	
	def thresholdEquation(self, r, density_threshold):
		
		diff = self.enclosedMass(r) / 4.0 / math.pi * 3.0 / r**3 - density_threshold
		
		return diff

	###############################################################################################

	# Return the spherical overdensity radius (in kpc / h) for a given mass definition and redshift. 

	def RDelta(self, z, mdef):
	
		density_threshold = densityThreshold(z, mdef)
		R = scipy.optimize.brentq(self.thresholdEquation, self.min_RDelta, self.max_RDelta, density_threshold)

		return R

	###############################################################################################

	# Return the spherical overdensity radius (in kpc / h) and mass (in Msun / h) for a given mass 
	# definition and redshift. 

	def RMDelta(self, z, mdef):
		
		R = self.RDelta(z, mdef)
		M = R_to_M(R, z, mdef)
		
		return R, M

	###############################################################################################

	# Return the spherical overdensity mass (in Msun / h) for a given mass definition and redshift.

	def MDelta(self, z, mdef):
		
		_, M = self.RMDelta(z, mdef)
		
		return M

###################################################################################################
# SPLINE DEFINED PROFILE
###################################################################################################

# This class takes an arbitrary array of radii and densities or enclosed masses as input, and 
# interpolates them using a splines (in log space). Note that there are three different ways of 
# specifying the density profile:
#
# density and mass:   Both density and mass are interpolated using splines. The consistency 
#                     between the two arrays is NOT checked! 
# density only:       In order for the enclosed mass to be defined, the density MUST be specified 
#                     all the way to r = 0. In that case, the mass is computed numerically, stored,
#                     and interpolated.
# mass only:          The density is computed as the derivative of the mass, stored, and
#                     interpolated.

class SplineDensityProfile(HaloDensityProfile):
	
	def __init__(self, r, rho = None, M = None):
		
		HaloDensityProfile.__init__(self)
		
		self.rmin = numpy.min(r)
		self.rmax = numpy.max(r)
		self.min_RDelta = self.rmin
		self.max_RDelta = self.rmax

		if rho == None and M == None:
			msg = 'Either mass or density must be specified.'
			raise Exception(msg)
		
		self.rho_spline = None
		self.M_spline = None
		logr = numpy.log(r)
		
		if M != None:
			logM = numpy.log(M)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if rho != None:
			logrho = numpy.log(rho)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		# Construct M(r) from density. For some reason, the spline integrator fails on the 
		# innermost bin, and the quad integrator fails on the outermost bin. 
		if self.M_spline == None:
			integrand = 4.0 * numpy.pi * r**2 * rho
			integrand_spline = scipy.interpolate.InterpolatedUnivariateSpline(r, integrand)
			logM = 0.0 * r
			for i in range(len(logM) - 1):
				logM[i], _ = scipy.integrate.quad(integrand_spline, 0.0, r[i])
			logM[-1] = integrand_spline.integral(0.0, r[-1])
			logM = numpy.log(logM)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if self.rho_spline == None:
			deriv = self.M_spline(numpy.log(r), nu = 1) * M / r
			logrho = numpy.log(deriv / 4.0 / numpy.pi / r**2)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		return

	###############################################################################################

	def density(self, r):

		return numpy.exp(self.rho_spline(numpy.log(r)))

	###############################################################################################
	
	def densityDerivativeLin(self, r):

		log_deriv = self.rho_spline(numpy.log(r), nu = 1)
		deriv = log_deriv * self.density(r) / r
		
		return deriv

	###############################################################################################

	def densityDerivativeLog(self, r):
	
		return self.rho_spline(numpy.log(r), nu = 1)
	
	###############################################################################################

	def enclosedMass(self, r):

		return numpy.exp(self.M_spline(numpy.log(r)))
	
###################################################################################################
# NFW PROFILE
###################################################################################################

# The Navarro-Frenk-White profile. The constructor accepts either the native parameters, central 
# density (in Msun h^2 / kpc^3) and scale radius (in kpc/h), or converts a mass and concentration 
# into the native NFW parameters.
	
class NFWProfile(HaloDensityProfile):

	def __init__(self, rhos = None, rs = None, \
				M = None, c = None, mdef = None, z = None):
		
		HaloDensityProfile.__init__(self)

		# The fundamental way to define an NFW profile by the central density and scale radius
		if rhos != None and rs != None:
			self.rhos = rhos
			self.rs = rs
			
		# Alternatively, the user can give a mass and concentration, together with mass definition
		# and redshift.
		elif M != None and c != None and mdef != None and z != None:
			self.rs = M_to_R(M, z, mdef) / c
			self.rhos = M / self.rs**3 / 4.0 / math.pi / self.mu(c)
		
		else:
			msg = 'An NFW profile must be define either using rhos and rs, or M, c, mdef, and z.'
			raise Exception(msg)
		
		return

	###############################################################################################

	# The density for an NFW profile with central density rhos (in Msun h^2 / kpc^3) and scale radius 
	# rs (in kpc / h), as a function of x = r / rs.
	
	def density(self, r):
	
		x = r / self.rs
		density = self.rhos / x / (1.0 + x)**2
		
		return density

	###############################################################################################

	def densityDerivativeLin(self, r):

		x = r / self.rs
		density_der = -self.rhos / self.rs * (1.0 / x**2 / (1.0 + x)**2 + 2.0 / x / (1.0 + x)**3)

		return density_der
	
	###############################################################################################

	def densityDerivativeLog(self, r):

		x = r / self.rs
		density_der = -(1.0 + 2.0 * x / (1.0 + x))

		return density_der
	
	###############################################################################################

	# The mu(c) function that appears in the expression for the enclosed mass in NFW profiles.
	
	def mu(self, c):
		
		mu = numpy.log(1.0 + c) - c / (1.0 + c)
		
		return mu

	###############################################################################################

	# The enclosed mass for an NFW profile with central density rhos (in Msun h^2 / kpc^3) and scale 
	# radius rs (in kpc / h), as a function of x = r / rs.
	
	def enclosedMass(self, r):
		
		x = r / self.rs
		mass = 4.0 * math.pi * self.rs**3 * self.rhos * self.mu(x)
		
		return mass
	
	###############################################################################################

	# The surface density (in units of Msun h / kpc^2).
	
	def surfaceDensity(self, r):
	
		x = r / self.rs
		
		if not Utilities.isArray(x):
			x_use = numpy.array([x])
		else:
			x_use = x
		
		surfaceDensity = 0.0 * x_use
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
		
			surfaceDensity[i] = 2.0 * self.rhos * self.rs / (x_use[i]**2 - 1.0) * fx
	
		if not Utilities.isArray(x):
			surfaceDensity = surfaceDensity[0]
	
		return surfaceDensity

	###############################################################################################

	# This equation is 0 when the enclosed density matches the given density_threshold.
		
	def thresholdEquation(self, r, density_threshold):
		
		x = r / self.rs
		diff = self.rhos * self.mu(x) * 3.0 / x**3 - density_threshold
		
		return diff

	###############################################################################################

	# Return the spherical overdensity radius (in kpc / h) for a given mass definition and redshift. 
	# This function is overwritten for the NFW profile as we have a better guess at the resulting
	# radius, namely the scale radius. Thus, the user can specify a minimum and maximum concentra-
	# tion that is considered.

	def RDelta(self, z, mdef, cmin = 0.1, cmax = 50.0):
	
		density_threshold = densityThreshold(z, mdef)
		R = scipy.optimize.brentq(self.thresholdEquation, self.rs * cmin, self.rs * cmax, density_threshold)

		return R

###################################################################################################
# DIEMER & KRAVTSOV 2014 PROFILE
###################################################################################################

# -------------------------------------------------------------------------------------------------
# This object specifies the parameters of the Diemer & Kravtsov 2014 halo density profile. The 
# full parameter freedom should almost never be used; the profile class below has functions to 
# compute the parameters. 
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
# -------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------------------------

class DK14Profile(HaloDensityProfile):
	
	def __init__(self, par = None, **kwargs):
	
		HaloDensityProfile.__init__(self)
		
		if par != None:
			self.par = par
		else:
			self.deriveParameters(**kwargs)

		self.accuracy_mass = 1E-4
		self.accuracy_radius = 1E-4

		return

	###############################################################################################

	def deriveParameters(self, M = None, c = None, z = None, mdef = None, \
			selected = 'by_mass', Gamma = None, part = 'both', be = None, se = None, \
			acc_warn = 0.01, acc_err = 0.05):
		
		# Declare shared variables; these parameters are advanced during the iterations
		par2 = {}
		par2['Rvir'] = 0.0
		par2['RDelta'] = 0.0
		par2['nu'] = 0.0
		self.par = DK14Parameters()
		
		RTOL = 0.01
		MTOL = 0.01
		GUESS_TOL = 2.5
		self.accuracy_mass = MTOL
		self.accuracy_radius = RTOL
		
		def radius_diff(R200m, par2, Gamma, rho_target, rho_vir, R_target):
			
			# Remember the parts we need to evaluate; this will get overwritten
			part_true = self.par.part
			self.par.R200m = R200m
			
			# Set nu_vir from previous Rvir
			Mvir = R_to_M(par2['Rvir'], z, 'vir')
			par2['nu'] = cosmo.M_to_nu(Mvir, z)
	
			# Set profile parameters
			self.par.alpha, self.par.beta, self.par.gamma, rt_R200m = \
				self.getFixedParameters(selected, nu = par2['nu'], z = z, Gamma = Gamma)
			self.par.rt = rt_R200m * R200m
	
			# Find rho_s; this can be done without iterating
			self.par.rho_s = 1.0
			self.par.part = 'inner'
			M200m = R_to_M(R200m, z, '200m')
			Mr_inner = self.enclosedMass(R200m, accuracy = MTOL)
			
			if part == 'both':
				self.par.part = 'outer'
				Mr_outer = self.enclosedMass(R200m, accuracy = MTOL)
			elif part == 'inner':
				Mr_outer = 0.0
			else:
				msg = "Invalid value for part, %s." % (part)
				raise Exception(msg)
				
			self.par.rho_s = (M200m - Mr_outer) / Mr_inner
			self.par.part = part_true
	
			# Now compute MDelta and Mvir from this new profile
			par2['RDelta'] = self.RDeltaLowlevel(par2['RDelta'], rho_target, guess_tolerance = GUESS_TOL)
			par2['Rvir'] = self.RDeltaLowlevel(par2['Rvir'], rho_vir, guess_tolerance = GUESS_TOL)
			
			return par2['RDelta'] - R_target
		
		# Test for wrong user input
		if part in ['outer', 'both'] and (be == None or se == None):
			msg = "Since part = %s, the parameters be and se must be set. The recommended values are 1.0 and 1.5." % (part)
			raise Exception(msg)
		
		# The user needs to set a cosmology before this function can be called
		cosmo = Cosmology.getCurrent()
		
		# Get concentration if the user hasn't supplied it, compute scale radius
		if c == None:
			c = HaloConcentration.concentration(M, mdef, z, statistic = 'median')
		R_target = M_to_R(M, z, mdef)
		par2['RDelta'] = R_target
		self.par.rs = R_target / c
		
		# Take a guess at nu_vir and R200m
		if mdef == 'vir':
			Mvir = M
		else:
			Mvir, par2['Rvir'], _ = changeMassDefinition(M, c, z, mdef, 'vir')
		par2['nu'] = cosmo.M_to_nu(Mvir, z)
		
		if mdef == '200m':
			R200m_guess = M_to_R(M, z, '200m')
		else:
			_, R200m_guess, _ = changeMassDefinition(M, c, z, mdef, '200m')
		
		# Iterate to find an M200m for which the desired mass is correct
		self.par.rho_m = cosmo.matterDensity(z)
		self.par.be = be
		self.par.se = se
		self.par.part = part
		rho_target = densityThreshold(z, mdef)
		rho_vir = densityThreshold(z, 'vir')
		args = par2, Gamma, rho_target, rho_vir, R_target
		self.par.R200m = scipy.optimize.brentq(radius_diff, R200m_guess / 1.3, R200m_guess * 1.3, \
							args = args, xtol = RTOL)
	
		# Check the accuracy of the result; M should be very close to MDelta now
		M_result = R_to_M(par2['RDelta'], z, mdef)
		err = (M_result - M) / M
		
		if abs(err) > acc_warn:
			msg = 'WARNING: DK14 profile parameters converged to an accuracy of %.1f percent.' % (abs(err) * 100.0)
			print(msg)
		if abs(err) > acc_err:
			msg = 'DK14 profile parameters not converged (%.1f percent error).' % (abs(err) * 100.0)
			raise Exception(msg)
		
		return

	###############################################################################################

	# Get the parameter values for the DK14 profile that should be fixed, or can be determined from the 
	# peak height or mass accretion rate. If selected is 'by_mass', only nu must be passed. If selected 
	# is 'by_accretion_rate', then both z and Gamma must be passed.
	
	def getFixedParameters(self, selected, nu = None, z = None, Gamma = None):
	
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

	###############################################################################################

	# The density of the DK14 profile as a function of radius (in kpc / h) and the profile 
	# parameters.
	
	def density(self, r):
		
		rho = 0.0 * r
		par = self.par
		
		if par.part in ['inner', 'both']:
			inner = par.rho_s * numpy.exp(-2.0 / par.alpha * ((r / par.rs) ** par.alpha - 1.0))
			fT = (1.0 + (r / par.rt) ** par.beta) ** (-par.gamma / par.beta)
			rho += inner * fT
		
		if par.part in ['outer', 'both']:
			outer = par.rho_m * (par.be * (r / 5.0 / par.R200m) ** (-par.se) + 1.0)
			rho += outer
		
		return rho

	###############################################################################################

	# The logarithmic slope of the density of the DK14 profile as a function of radius (in kpc / h) 
	# and the profile parameters.
	
	def densityDerivativeLin(self, r):
		
		rho = 0.0 * r
		drho_dr = 0.0 * r
		par = self.par
		
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
		
		return drho_dr

	###############################################################################################

	# The logarithmic slope of the density of the DK14 profile as a function of radius (in kpc / h) 
	# and the profile parameters.
	
	def densityDerivativeLog(self, r):
		
		drho_dr = self.densityDerivativeLin(r)
		rho = self.density(r)
		der = drho_dr * r / rho
		
		return der

	###############################################################################################

	# Low-level function to compute a spherical overdensity radius given the parameters of a DK14 
	# profile, the desired overdensity threshold, and an initial guess. A more user-friendly version
	# can be found above (DK14_getMR).
	
	def RDeltaLowlevel(self, R_guess, density_threshold, guess_tolerance = 5.0):
			
		R = scipy.optimize.brentq(self.thresholdEquation, R_guess / guess_tolerance, \
				R_guess * guess_tolerance, args = density_threshold, xtol = self.accuracy_radius)
		
		return R
	
	###############################################################################################

	# This function returns the spherical overdensity radius (in kpc / h) given a mass definition
	# and redshift. We know R200m and thus M200m for a DK14 profile, and use those parameters to
	# compute what R would be for an NFW profile and use this radius as an initial guess.
	
	def RDelta(self, z, mdef):
	
		M200m = R_to_M(self.par.R200m, z, mdef)
		density_threshold = densityThreshold(z, mdef)
		_, R_guess, _ = changeMassDefinition(M200m, self.par.R200m / self.par.rs, z, '200m', mdef)
		R = self.RDeltaLowlevel(R_guess, density_threshold)
	
		return R

###################################################################################################
# FUNCTIONS THAT ARE INDEPENDENT OF THE FORM OF THE DENSITY PROFILE
###################################################################################################

# This function returns a spherical overdensity halo radius in kpc / h for a halo mass M in Msun/h,
# a redshift z, and a mass definition (see the documentation at the top of this file for valid mass 
# definition formats.)

def M_to_R(M, z, mdef):

	rho = densityThreshold(z, mdef)
	R = (M * 3.0 / 4.0 / math.pi / rho)**(1.0 / 3.0)

	return R

###################################################################################################

# This function returns a spherical overdensity halo mass in Msun / h for a halo radius R in kpc/h,
# a redshift z, and a mass definition (see the documentation at the top of this file for valid mass 
# definition formats.)

def R_to_M(R, z, mdef):

	rho = densityThreshold(z, mdef)
	M = 4.0 / 3.0 * math.pi * rho * R**3

	return M

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

def densityThreshold(z, mdef):

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

# The halo bias, using the approximation of Tinker et al. 2010, ApJ 724, 878. The mass definition,
# mdef, MUST correspond to the mass that was used to evaluate nu. Note that the Tinker bias 
# function is universal in redshift at fixed peak height, but only for mass definitions defined 
# wrt the mean density of the universe. For other definitions, Delta_m evolves with redshift, 
# leading to an evolving bias at fixed peak height. 

def haloBiasFromNu(nu, z, mdef):

	cosmo = Cosmology.getCurrent()
	Delta = densityThreshold(z, mdef) / cosmo.matterDensity(z)
	y = numpy.log10(Delta)

	A = 1.0 + 0.24 * y * numpy.exp(-1.0 * (4.0 / y)**4)
	a = 0.44 * y - 0.88
	B = 0.183
	b = 1.5
	C = 0.019 + 0.107 * y + 0.19 * numpy.exp(-1.0 * (4.0 / y)**4)
	c = 2.4

	bias = 1.0 - A * nu**a / (nu**a + Cosmology.AST_delta_collapse**a) + B * nu**b + C * nu**c

	return bias

###################################################################################################

# Wrapper function for the function above which accepts M instead of nu.

def haloBias(M, z, mdef):
	
	cosmo = Cosmology.getCurrent()
	nu = cosmo.M_to_nu(M, z)
	b = haloBiasFromNu(nu, z, mdef)
	
	return b

###################################################################################################
# FUNCTIONS THAT CAN REFER TO DIFFERENT FORMS OF THE DENSITY PROFILE
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
# another. Use the changMassDefinition() function below for this case. See the documentation at 
# the top of this file for valid mass definition formats.

def pseudoEvolve(M_i, c_i, z_i, mdef_i, z_f, mdef_f, profile = 'nfw'):

	if profile == 'nfw':

		prof = NFWProfile(M = M_i, c = c_i, z = z_i, mdef = mdef_i)
		Rnew = prof.RDelta(z_f, mdef_f)
		cnew = Rnew / prof.rs
	
	elif profile == 'dk14':
		
		prof = DK14Profile(M = M_i, mdef = mdef_i, z = z_i, c = c_i, selected = 'by_mass', part = 'inner')
		
		if mdef_f == '200m':
			Rnew = prof.par.R200m
		else:
			Rnew = prof.RDelta(z_f, mdef_f)
		cnew = Rnew / prof.rs
		
	else:
		msg = 'This function is not defined for profile %s.' % (profile)
		raise Exception(msg)

	Mnew = R_to_M(Rnew, z_f, mdef_f)
	
	return Mnew, Rnew, cnew

###################################################################################################

# Get the spherical overdensity mass, radius and concentration for a mass definition mdef_out, 
# given the mass and concentration at redshift z and for mass definition mdef_in. See the 
# documentation at the top of this file for valid mass definition formats.
#
# The function needs to assume a form of the density profile as a function of M and c. By default,
# the NFW profile is used, but the user can also choose profile = 'dk14' for the DK14 profile. Note
# that the latter version is much slower. If return_parameters == True, the function also returns
# additional information about the density profile used for the conversion. 

def changeMassDefinition(M, c, z, mdef_in, mdef_out, profile = 'nfw'):

	return pseudoEvolve(M, c, z, mdef_in, z, mdef_out, profile = profile)

###################################################################################################
