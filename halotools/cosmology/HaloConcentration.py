###################################################################################################
#
# Concentration.py 		(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# This module implements the concentration model of Diemer & Kravtsov 2014b, as well as several 
# other, mostly power-law based models. The main function in this module, concentration(), is a 
# wrapper for all models and mass definitions. Alternatively, the user can also call the individual
# model functions directly. Note that most models are only valid over a certain range of masses, 
# redshifts, and cosmology.
#
# Beside the general concentration() function, the user can directly call the specific functions
# for each model. In the case of the DK14 model, for example, that might make sense if the user 
# knows the peak height of halos rather than their mass. See the documentation of the functions
# below.
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

# Concentration as a function of halo mass, for different concentration models, statistics, and 
# conversion profiles. For some models, a cosmology must be set (see the documentation of the 
# Cosmology.py module). The parameters are:
#
# -------------------------------------------------------------------------------------------------
# Parameter        Explanation
# -------------------------------------------------------------------------------------------------
# M                Halo mass(es) in units of Msun / h; can be a number or a numpy array.
# mdef             The mass definition in which the halo mass M is given, and in which c is
#                  returned. Mass definitions are given in a string format. Valid inputs are:
#
#                  vir     A varying overdensity mass definition, implemented using the fitting 
#                          formula of Bryan & Norman 1998
#                  ****c   An integer number times the critical density of the universe, e.g. 200c
#                  ****m   An integer number times the matter density of the universe, e.g. 200m
#
# z                Redshift
# model            The model of the c-M relation used; see list below.
# statistic        'mean' or 'median' concentration. Note that many models do not distinguish
#                  between these statistics. For those models, this parameter is ignored.
# conversion_prof. The profile form used to convert from one mass definition to another. See 
#                  explanation below.
# range_return     If True, the function returns a boolean mask indicating the validty of the 
#                  returned concentrations (see return value below).
# range_warning    If True, a warning is thrown if the user requested a mass or redshift where the 
#                  model is not calibrated. Note that no such warning is thrown if the cosmology
#                  does not match that of the model!
#
# -------------------------------------------------------------------------------------------------
# Return value     Explanation
# -------------------------------------------------------------------------------------------------
# c                Halo concentration(s) in the mass definition mdef. Depending on the input M, 
#                  this can be a number or a numpy array.
# mask (optional)  If range_return == True, the function also returns a numpy array of True/False 
#                  values, where False indicates that the model was not calibrated at the chosen 
#                  mass or redshift.
#
# -------------------------------------------------------------------------------------------------
# Model          Mass defs.       Mass(z=0)        Redshift   Cosmo.   Reference
# -------------------------------------------------------------------------------------------------
# dk14           200c             All              All        All      Diemer & Kravtsov 2014
# dutton14       200c, vir        M > 1E10         0 < z < 5  Planck1  Dutton & Maccio 2014, MNRAS 441, 3359
# bhattacharya13 200c, vir, 200m  2E12 < M < 2E15  0 < z < 2  WMAP7    Bhattacharya et al. 2013, ApJ 766, 32
# prada12        200c             All              All        All      Prada et al. 2012, MNRAS 423, 3018
# klypin11       vir              3E10 < M < 5E14  0          WMAP7    Klypin et al. 2011, ApJ 740, 102
# duffy08        200c, vir, 200m  1E11 < M < 1E15  0 < z < 2  WMAP5    Duffy et al. 2008, MNRAS 390, L64
# -------------------------------------------------------------------------------------------------
#
# If the user requests a mass definition that is not one of the native definitions of the c-M model,
# the mass and concentration are converted, necessarily assuming a particular form of the density
# profile. For this purpose, the user can choose between 'nfw' and 'dk14' profiles.
#
# WARNING 1:    The conversion to other mass definitions degrades the accuracy of the predicted
#               concentration. We have evaluated this added inaccuracy for the DK14 concentration 
#               model (see Appendix C in Diemer & Kravtsov 2014b for details). Compared to our 
#               simulation results, the accuracy is degraded as follows:
#
#               c200c    ~5-10% (nominal accuracy of the model)
#               c500c    ~5-10% (nu < 2), ~20% (nu > 2, NFW conversion), 
#                        ~15% (nu > 2, DK14 conversion)
#               cvir     ~5-10% (z > 1, or nu < 2), ~10-20% (nu > 2, z < 1, NFW conversion), 
#                        ~10-20% (nu > 2, z < 1, DK14 conversion)
#               c200m    ~10% (z > 1, or nu < 2), ~15-20% (nu > 2, z < 1, NFW conversion), 
#                        ~10-15% (nu > 2, z < 1, DK14 conversion)
#
# WARNING 2:    As the above results show, using the DK14 profile for the conversion gives slightly 
#               more accurate results, but is also significantly slower.
# -------------------------------------------------------------------------------------------------

def concentration(M, mdef, z, \
				model = 'dk14', statistic = 'median', conversion_profile = 'nfw', \
				range_return = False, range_warning = False):
	
	# Evaluate the concentration model
	def evaluateC(func, M, limited, args):
		if limited:
			c, mask = func(M, *args)
		else:
			mask = None
			c = func(M, *args)
		return c, mask
	
	# This equation is zero for a mass MDelta (in the mass definition of the c-M model) when the
	# corresponding mass in the user's mass definition is M_desired.
	def eq(MDelta, M_desired, mdef_model, func, limited, args):
		cDelta, _ = evaluateC(func, MDelta, limited, args)
		Mnew, _, _ = HaloDensityProfile.convertMassDefinition(MDelta, cDelta, z, mdef_model, mdef,\
												profile = 'nfw')
		return Mnew - M_desired

	# Distinguish between models
	if model == 'dk14':
		mdefs_model = ['200c']
		func = dk14_c200c_M
		args = (z, statistic)
		limited = False
		
	elif model == 'dutton14':
		mdefs_model = ['200c', 'vir']
		func = dutton14_c
		args = (z,)
		limited = True

	elif model == 'bhattacharya13':
		mdefs_model = ['200c', 'vir', '200m']
		func = bhattacharya13_c
		args = z,
		limited = True

	elif model == 'prada12':
		mdefs_model = ['200c']
		func = prada12_c200c
		args = z,
		limited = False

	elif model == 'klypin11':
		mdefs_model = ['vir']
		func = klypin11_cvir
		args = z,
		limited = True
		
	elif model == 'duffy08':
		mdefs_model = ['200c', 'vir', '200m']
		func = duffy08_c
		args = z,
		limited = True
	
	else:
		msg = 'Unknown model, %s.' % (model)
		raise Exception(msg)

	# Now check whether the definition the user has requested is the native definition of the model.
	# If yes, we just return the model concentration. If not, the problem is much harder. Without 
	# knowing the concentration, we do not know what mass in the model definition corresponds to 
	# the input mass M. Thus, we need to find both M and c iteratively.
	if mdef in mdefs_model:
		
		if len(mdefs_model) > 1:
			args = args + (mdef,)
		c, mask = evaluateC(func, M, limited, args)
		
	else:
		if not Utilities.isArray(M):
			M_use = numpy.array([M])
		else:
			M_use = M
		
		mdef_model = mdefs_model[0]
		if len(mdefs_model) > 1:
			args = args + (mdef_model,)
		
		c = 0.0 * M_use
		mask = numpy.array([True] * len(M_use))
		for i in range(len(M_use)):
			M_min = 0.01 * M_use[i]
			M_max = 100.0 * M_use[i]
			args_solver = M_use[i], mdef_model, func, limited, args
			MDelta = scipy.optimize.brentq(eq, M_min, M_max, args = args_solver)
			cDelta, mask[i] = evaluateC(func, MDelta, limited, args)
			_, _, c[i] = HaloDensityProfile.convertMassDefinition(MDelta, cDelta, z, mdef_model, \
										mdef, profile = conversion_profile)
		if not Utilities.isArray(M):
			c = c[0]
			mask = mask[0]

	# If there are no limits on the model, mask is an array of True
	if not limited and range_return:
		mask = numpy.array([True] * len(M))
	
	# Spit out warning if the range was violated
	if range_warning and limited:
		if False in mask:
			raise Warning('Some masses or redshifts are outside the validity of the concentration model.')
	
	if range_return:
		return c, mask
	else:
		return c

###################################################################################################
# DIEMER & KRAVTSOV 2014 MODEL
###################################################################################################
#
# The functions in this section include:
#
# dk14_c200c_M  Get c200c given a mass M200c (or an array of M200c) at a certain redshift. The 
#               power spectrum slope and peak height are computed internally. For this purpose, a
#               cosmology must be set (see the documentation of Cosmology.py).
# 
# dk14_c200c_nu As c200c_M, but takes peak height instead of mass as an input parameter. This peak
#               height must be computed with the tophat filter and M = M200c.
#
# dk14_c200c_n  The universal concentration for a given power spectrum slope, n, and peak height, 
#               nu. Given these parameters, the result is independent of redshift. This function
#               is mostly intended for internal use, but can be used to predict concentration at
#               a fixed power spectrum slope.
#
###################################################################################################

# Model constants

dk14_kappa = 0.69

dk14_median_phi_0 = 6.58
dk14_median_phi_1 = 1.37
dk14_median_eta_0 = 6.82
dk14_median_eta_1 = 1.42
dk14_median_alpha = 1.12
dk14_median_beta = 1.69

dk14_mean_phi_0 = 7.14
dk14_mean_phi_1 = 1.60
dk14_mean_eta_0 = 4.10
dk14_mean_eta_1 = 0.75
dk14_mean_alpha = 1.40
dk14_mean_beta = 0.67

###################################################################################################

# The prediction of our model for a given mass M200c, redhsift, and statistic. For other mass 
# definitions, see the more general concentration() function below.

def dk14_c200c_M(M200c, z, statistic = 'median'):
	
	cosmo = Cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * M200c / M200c
	else:
		n = dk14_compute_n_M(M200c, z)
	
	nu = cosmo.M_to_nu(M200c, z)
	ret = dk14_c200c_n(nu, n, statistic)

	return ret

###################################################################################################

# The prediction of our model for a given peak height, redhsift, and statistic. Our model is
# calibrated for peak heights computed using the tophat filter, thus the user cannot change the 
# filter. Note that this function returns c200c, and expects that the peak height was computed for 
# a halo mass M200c. For other mass definitions, see the more general concentration() function 
# below.

def dk14_c200c_nu(nu200c, z, statistic = 'median'):
	
	cosmo = Cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * nu200c / nu200c
	else:
		n = dk14_compute_n_nu(nu200c, z)
	
	ret = dk14_c200c_n(nu200c, n, statistic)

	return ret

###################################################################################################

# The universal prediction of our model for a given peak height, power spectrum slope, and
# statistic.

def dk14_c200c_n(nu, n, statistic = 'median'):

	if statistic == 'median':
		floor = dk14_median_phi_0 + n * dk14_median_phi_1
		nu0 = dk14_median_eta_0 + n * dk14_median_eta_1
		alpha = dk14_median_alpha
		beta = dk14_median_beta
	elif statistic == 'mean':
		floor = dk14_mean_phi_0 + n * dk14_mean_phi_1
		nu0 = dk14_mean_eta_0 + n * dk14_mean_eta_1
		alpha = dk14_mean_alpha
		beta = dk14_mean_beta
	else:
		raise Exception("Unknown statistic.")
	
	c = 0.5 * floor * ((nu0 / nu)**alpha + (nu / nu0)**beta)
	
	return c

# Compute the characteristic wavenumber for a particular halo mass.

def dk14_wavenumber_k_R(M):

	cosmo = Cosmology.getCurrent()
	rho0 = cosmo.matterDensity(0.0)
	R = (3.0 * M / 4.0 / math.pi / rho0) ** (1.0 / 3.0) / 1000.0
	k_R = 2.0 * math.pi / R * dk14_kappa

	return k_R

###################################################################################################

# Get the slope n = d log(P) / d log(k) at a scale k_R and a redshift z. The slope is computed from
# the Eisenstein & Hu 1998 approximation to the power spectrum (without BAO).

def dk14_compute_n(k_R, z):

	if numpy.min(k_R) < 0:
		raise Exception("k_R < 0.")

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

# Wrapper for the function above which accepts M instead of k.

def dk14_compute_n_M(M, z):

	k_R = dk14_wavenumber_k_R(M)
	n = dk14_compute_n(k_R, z)
	
	return n

###################################################################################################

# Wrapper for the function above which accepts nu instead of M.

def dk14_compute_n_nu(nu, z):

	cosmo = Cosmology.getCurrent()
	M = cosmo.nu_to_M(nu, z)
	n = dk14_compute_n_M(M, z)
	
	return n

###################################################################################################
# DUTTON & MACCIO 2014 MODEL
###################################################################################################

# The power-law fits of Dutton & Maccio 2014, MNRAS 441, 3359. This model was calibrated for the 
# Planck 1-year cosmology.

def dutton14_c(M, z, mdef):

	if mdef == '200c':
		a = 0.520 + (0.905 - 0.520) * numpy.exp(-0.617 * z**1.21)
		b = -0.101 + 0.026 * z
	elif mdef == 'vir':
		a = 0.537 + (1.025 - 0.537) * numpy.exp(-0.718 * z**1.08)
		b = -0.097 + 0.024 * z
	else:
		msg = 'Invalid mass definition for Dutton & Maccio 2014 model, %s.' % mdef
		raise Exception(msg)
	
	logc = a + b * numpy.log10(M / 1E12)
	c = 10**logc

	mask = (M > 1E10) & (z <= 5.0)

	return c, mask

###################################################################################################
# BHATTACHARYA ET AL 2013 MODEL
###################################################################################################

# The model of Bhattacharya et al. 2013, ApJ 766, 32, a fitting function to the relation between
# concentration and peak height. This model was calibrated for a WMAP7 cosmology.

def bhattacharya13_c(M, z, mdef):

	cosmo = Cosmology.getCurrent()
	D = cosmo.growthFactor(z)
	
	# Note that peak height in the B13 paper is defined wrt. the mass definition in question, so 
	# we can just use M to evaluate nu. 
	nu = cosmo.M_to_nu(M, z)

	if mdef == '200c':
		c_fit = 5.9 * D**0.54 * nu**-0.35
	elif mdef == 'vir':
		c_fit = 7.7 * D**0.90 * nu**-0.29
	elif mdef == '200m':
		c_fit = 9.0 * D**1.15 * nu**-0.29
	else:
		msg = 'Invalid mass definition for Bhattacharya et al. 2013 model, %s.' % mdef
		raise Exception(msg)
				
	M_min = 2E12
	M_max = 2E15
	if z > 0.5:
		M_max = 2E14
	if z > 1.5:
		M_max = 1E14
	mask = (M >= M_min) & (M <= M_max) & (z <= 2.0)
	
	return c_fit, mask

###################################################################################################
# PRADA ET AL 2012 MODEL
###################################################################################################

# The model of Prada et al. 2012, MNRAS 423, 3018. Like our model, this model predicts c200c and is
# based on the c-nu relation. The model was calibrated on the Bolshoi and Multidark simulations, 
# but is in principle applicable to any cosmology. The implementation follows equations 12 to 22 in 
# Prada et al. 2012. This function uses the exact values for sigma rather than their approximation 
# This makes a small difference.

def prada12_c200c(M200c, z):

	def cmin(x):
		return 3.681 + (5.033 - 3.681) * (1.0 / math.pi * math.atan(6.948 * (x - 0.424)) + 0.5)
	def smin(x):
		return 1.047 + (1.646 - 1.047) * (1.0 / math.pi * math.atan(7.386 * (x - 0.526)) + 0.5)

	cosmo = Cosmology.getCurrent()
	nu = cosmo.M_to_nu(M200c, z)

	a = 1.0 / (1.0 + z)
	x = (cosmo.OL0 / cosmo.Om0) ** (1.0 / 3.0) * a
	B0 = cmin(x) / cmin(1.393)
	B1 = smin(x) / smin(1.393)
	temp_sig = 1.686 / nu
	temp_sigp = temp_sig * B1
	temp_C = 2.881 * ((temp_sigp / 1.257) ** 1.022 + 1) * numpy.exp(0.06 / temp_sigp ** 2)
	c200c = B0 * temp_C

	return c200c

###################################################################################################
# KLYPIN ET AL 2011 MODEL
###################################################################################################

# The power-law fit of Klypin et al. 2011, ApJ 740, 102. This model was calibrated for the WMAP7
# cosmology of the Bolshoi simulation. Note that this model relies on concentrations that were
# measured approximately from circular velocities, rather than from a fit to the actual density 
# profiles. 
#
# Klypin et al. 2011 also give fits at particular redshifts other than zero. However, there is no 
# clear procedure to interpolate between redshifts, particularly since the z = 0 relation has a 
# different functional form than the high-z relations. Thus, we only implement the z = 0 relation 
# here.  

def klypin11_cvir(Mvir, z):

	cvir = 9.6 * (Mvir / 1E12)**-0.075
	mask = (Mvir > 3E10) & (Mvir < 5E14) & (z < 0.01)

	return cvir, mask

###################################################################################################
# DUFFY ET AL 2008 MODEL
###################################################################################################

# The power-law fits of Duffy et al. 2008, MNRAS 390, L64. This model was calibrated for a WMAP5
# cosmology.

def duffy08_c(M, z, mdef):
	
	if mdef == '200c':
		A = 5.71
		B = -0.084
		C = -0.47
	elif mdef == 'vir':
		A = 7.85
		B = -0.081
		C = -0.71
	elif mdef == '200m':
		A = 10.14
		B = -0.081
		C = -1.01
	else:
		msg = 'Invalid mass definition for Duffy et al. 2008 model, %s.' % mdef
		raise Exception(msg)

	c = A * (M / 2E12)**B * (1.0 + z)**C
	mask = (M >= 1E11) & (M <= 1E15) & (z <= 2.0)
	
	return c, mask

###################################################################################################
