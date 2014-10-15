###################################################################################################
#
# Concentration.py 		(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# This module implements the concentration model of Diemer & Kravtsov 2014b. Internally, this model
# is based in concentration-peak height space, but the user can give either mass or peak height as
# input parameters. There are multiple types of concentration functions in this module, depending
# on the desired overdensity definition:
# 
# c200c_M       Get c200c given a mass M200c (or an array of M200c) at a certain redshift. The 
#               power spectrum slope and peak height are computed internally. For this purpose, a
#               cosmology must be set (see the documentation of Cosmology.py).
# 
# c200c_nu      As c200c_M, but takes peak height instead of mass as an input parameter. This peak
#               height must be computed with the tophat filter and M = M200c.
#
# c200c_n		The universal concentration for a given power spectrum slope, n, and peak height, 
#               nu. Given these parameters, the result is independent of redshift. This function
#               is mostly intended for internal use, but can be used to predict concentration at
#               a fixed power spectrum slope.
#
# WARNING:      The following concentration() function is not as accurate as the c200c() functions
#               above, simply because it uses NFW profiles to convert from M200c and c200c to 
#               other definitions. Real profiles do not exactly follow the NFW form (see, e.g.,
#               Diemer & Kravtsov 2014a), leading to inaccuracies of up to 20% at some redshifts
#               and masses. See Appendix C in Diemer & Kravtsov 2014b for details.
#
# concentration This function predicts the c-M relation for mass definitions other than 200c.
#               Mass definitions are given in a string format. Valid inputs are 
#
#               vir     A varying overdensity mass definition, implemented using the fitting 
#                       formula of Bryan & Norman 1998.
#               ****c   An integer number times the critical density of the universe, such as 200c. 
#               ****m   An integer number times the matter density of the universe, such as 200m. 
#
#               The concentration returned is in the same mass definition as the input mass. Note 
#               that this function is slower than the c200c functions because the concentration 
#               needs to be determined iteratively.
#
###################################################################################################

import math
import numpy
import scipy.interpolate
import scipy.optimize
import Utilities
import Cosmology
import HaloDensityProfile

###################################################################################################
# MODEL CONSTANTS
###################################################################################################

def_kappa = 0.69

def_median_phi_0 = 6.58
def_median_phi_1 = 1.37
def_median_eta_0 = 6.82
def_median_eta_1 = 1.42
def_median_alpha = 1.12
def_median_beta = 1.69

def_mean_phi_0 = 7.14
def_mean_phi_1 = 1.60
def_mean_eta_0 = 4.10
def_mean_eta_1 = 0.75
def_mean_alpha = 1.40
def_mean_beta = 0.67

###################################################################################################
# CONCENTRATION
###################################################################################################

# The prediction of our model for a given mass M200c, redhsift, and statistic. For other mass 
# definitions, see the more general concentration() function below.

def c200c_M(M200c, z, statistic = 'median'):
	
	cosmo = Cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * M200c / M200c
	else:
		n = compute_n_M(M200c, z)
	
	nu = cosmo.M_to_nu(M200c, z)
	ret = c200c_n(nu, n, statistic)

	return ret

###################################################################################################

# The prediction of our model for a given peak height, redhsift, and statistic. Our model is
# calibrated for peak heights computed using the tophat filter, thus the user cannot change the 
# filter. Note that this function returns c200c, and expects that the peak height was computed for 
# a halo mass M200c. For other mass definitions, see the more general concentration() function 
# below.

def c200c_nu(nu200c, z, statistic = 'median'):
	
	cosmo = Cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * nu200c / nu200c
	else:
		n = compute_n_nu(nu200c, z)
	
	ret = c200c_n(nu200c, n, statistic)

	return ret

###################################################################################################

# The universal prediction of our model for a given peak height, power spectrum slope, and
# statistic.

def c200c_n(nu, n, statistic = 'median'):

	if statistic == 'median':
		floor = def_median_phi_0 + n * def_median_phi_1
		nu0 = def_median_eta_0 + n * def_median_eta_1
		alpha = def_median_alpha
		beta = def_median_beta
	elif statistic == 'mean':
		floor = def_mean_phi_0 + n * def_mean_phi_1
		nu0 = def_mean_eta_0 + n * def_mean_eta_1
		alpha = def_mean_alpha
		beta = def_mean_beta
	else:
		raise Exception("Unknown statistic.")
	
	c = 0.5 * floor * ((nu0 / nu)**alpha + (nu / nu0)**beta)
	
	return c

###################################################################################################

# WARNING: The concentration predicted for mass definitions other than c200c is not as accurate as
#          for c200c, due to differences between real profiles and the NFW approximation. See the 
#          top of this file for details. Whenever possible, it is recommended to use the c200c 
#          functions above.

# Concentration for general mass definitions. In contrast to the c200c functions above, the input 
# mass M can be in any mass definition mdef (see documentation at the top of this file for valid
# formats). The concentration is returned in the same mass definition, e.g. cvir if mdef == 'vir'.

def concentration(M, z, statistic = 'median', mdef = '200c', conversion_profile = 'nfw'):
	
	def eq(M200c, M_desired):
		
		c200c = c200c_M(M200c, z, statistic = statistic)
		Mnew, _, _ = HaloDensityProfile.convertMassDefinition(M200c, c200c, z, '200c', mdef)
		
		return Mnew - M_desired
	
	if mdef == '200c':
		
		# This is the easy case, we just return the c200c function above.
		cosmo = Cosmology.getCurrent()
		nu200c = cosmo.M_to_nu(M, z)
		c = c200c_nu(nu200c, z, statistic = statistic)
		
	else:
		
		# This case is much harder. Without knowing the concentration, we do not know what M200c
		# corresponds to the input mass M. Thus, we need to find M and c iteratively.
		cosmo = Cosmology.getCurrent()
		
		if not Utilities.isArray(M):
			M_use = numpy.array([M])
		else:
			M_use = M
		
		c = 0.0 * M_use
		for i in range(len(M_use)):
			args = M_use[i]
			M_min = 0.01 * M_use[i]
			M_max = 100.0 * M_use[i]
			M200c = scipy.optimize.brentq(eq, M_min, M_max, args)
			c200c = c200c_M(M200c, z, statistic = statistic)
			_, _, c[i] = HaloDensityProfile.convertMassDefinition(M200c, c200c, z, '200c', mdef, \
												profile = conversion_profile)
			
		if not Utilities.isArray(M):
			c = c[0]
		
	return c

###################################################################################################
# HELPER FUNCTIONS (FOR INTERNAL USE)
###################################################################################################

# Compute the characteristic wavenumber for a particular halo mass.

def wavenumber_k_R(M):

	cosmo = Cosmology.getCurrent()
	rho0 = cosmo.matterDensity(0.0)
	R = (3.0 * M / 4.0 / math.pi / rho0) ** (1.0 / 3.0) / 1000.0
	k_R = 2.0 * math.pi / R * def_kappa

	return k_R

###################################################################################################

# Get the slope n = d log(P) / d log(k) at a scale k_R and a redshift z. The slope is computed from
# the Eisenstein & Hu 1998 approximation to the power spectrum (without BAO).

def compute_n(k_R, z):

	if numpy.min(k_R) < 0:
		raise Exception("getPkSlope: ERROR: k_R < 0.")

	cosmo = Cosmology.getCurrent()

	# We need coverage to compute the local slope at kR. For the spline, we evaluate a somewhat
	# larger range in k.
	k_min = numpy.min(k_R) / 2.0
	k_max = numpy.max(k_R) * 2.0
	
	# Now compute a grid of k and P(k) values
	k = 10**numpy.arange(numpy.log10(k_min), numpy.log10(k_max), 0.01)
	Pk = cosmo.matterPowerSpectrum(k, Pk_source = 'eh98smooth')
	
	# Compute the slope
	logk = numpy.log(k)
	a, b, c = scipy.interpolate.splrep(logk, numpy.log(Pk), s = 0.0)
	tup = a, b, c
	dPdk = scipy.interpolate.splev(logk, tup, der = 1)
	n = numpy.interp(k_R, k, dPdk)
	
	return n

###################################################################################################

# Wrapper for the function above which takes M instead of k.

def compute_n_M(M, z):

	k_R = wavenumber_k_R(M)
	n = compute_n(k_R, z)
	
	return n

###################################################################################################

# Wrapper for the function above which takes nu instead of M.

def compute_n_nu(nu, z):

	cosmo = Cosmology.getCurrent()
	M = cosmo.nu_to_M(nu, z)
	n = compute_n_M(M, z)
	
	return n

###################################################################################################
