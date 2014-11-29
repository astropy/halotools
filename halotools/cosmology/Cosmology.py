###################################################################################################
#
# Cosmology.py 		(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# This module is an implementation of the LambdaCDM cosmology, with a focus on structure formation 
# applications. It has no dependencies except for standarad python libraries such as numpy and 
# scipy, and requires no installation (simply use "import Cosmology").
#
# -------------------------------------------------------------------------------------------------
# FUNCTIONALITY
# -------------------------------------------------------------------------------------------------
#
# The Cosmology class below implements a standard Lambda-CDM cosmology, with fixed dark energy 
# equation of state (w = const), and ignoring the constribution of relativistic species (photons 
# and neutrinos). This implementation is focused on structure formation applications. The most 
# important functions are:
# 
# Ez(), Om(), age() etc.	Standard cosmology quantities: E(z), distances, times, and densities 
#                         	as a function of redshift
# growthFactor()			The linear growth factor D+(z)
# matterPowerSpectrum()		The linear matter power spectrum from the Eisenstein & Hu 98 
#  							approximation, and its logarithmic derivative
# sigma(), M_to_nu() etc.   The variance of the linear density field, its derivative d log(sigma) 
#							/ d log(R), peak height, and non-linear mass M*.
# correlationFunction()		The linear matter-matter correlation function
# peakCurvature()			The curvature of the peaks in a Gaussian random field.
#
# Unless otherwise stated, all functions of redshift, mass, radius or wavenumber can take both 
# individual numbers or arrays as input. See the documentation of the individual functions below.
#
# -------------------------------------------------------------------------------------------------
# BASIC USAGE
# -------------------------------------------------------------------------------------------------
#
# Create a cosmology object using a named set, e.g.
#
# cosmo = Cosmology.setCosmology('planck1')
#
# See below for a list of named cosmologies. Parameters from this standard cosmology can be 
# overwritten with additional parameters, e.g.
#                 
# cosmo = Cosmology.setCosmology('planck1', {"interpolation": False})
#
# Alternatively, the user can create an entirely new cosmology with a parameter dictionary similar 
# to those below. Only the main cosmological parameters are mandatory, all other parameters can be
# left to their default values:
#
# params = {'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
# cosmo = Cosmology.setCosmology('myCosmo', params)
#
# Whichever way a cosmology is set, the current cosmology is also stored as a global variable so 
# that the user can obtain it at any time using
#
# cosmo = Cosmology.getCurrent()
#
# The user can change cosmological parameters at run-time, but MUST call the update function 
# directly after the changes. This function ensures that the parameters are consistent 
# (e.g., flatness), and discards pre-computed quantities:
#
# cosmo.Om0 = 0.31
# cosmo.checkForChangedCosmology()
# 
# -------------------------------------------------------------------------------------------------
# PERFORMANCE OPTIMIZATION
# -------------------------------------------------------------------------------------------------
#
# This module is optimized for fast performance, particularly in computationally intensive
# functions such as the correlation function. All computationally intensive quantities are, by 
# default, tabulated, stored in files, and re-loaded when the same cosmology is loaded again. 
#
# For some, rare applications, the user might want to turn this behavior off. Please see the 
# documentation of the 'interpolation' and 'storage' parameters to the Cosmology() class.
#
# -------------------------------------------------------------------------------------------------
# WARNING
# -------------------------------------------------------------------------------------------------
#
# While this unit has been tested against various other codes, there is no guarantee of that it is 
# bug-free. Use at your own risk.
#
###################################################################################################

import os
import math
import numpy
import scipy.integrate
import scipy.special
import scipy.interpolate
import hashlib
import pickle
import Utilities

###################################################################################################
# Useful constants
###################################################################################################

# Cosmological distances in cm
AST_pc  = 3.08568025E18
AST_kpc = 3.08568025E21
AST_Mpc = 3.08568025E24

# A year in seconds
AST_year = 31556926.0

# Msun in g
AST_Msun = 1.98892E33

# Speed of light in cm / s
AST_c = 2.99792458E10

# G in kpc km^2 / M_sun / s^2
AST_G = 4.30172E-6

# Critical density at z = 0 in units of Msun h^2 / kpc^3 and Mpc^3
AST_rho_crit_0_kpc3 = 2.774848e+02
AST_rho_crit_0_Mpc3 = 2.774848e+11

# Critical collapse overdensity
AST_delta_collapse = 1.686

###################################################################################################
# Global variable for cosmology object
###################################################################################################

current_cosmo = None

###################################################################################################
# Cosmologies
###################################################################################################

# The following named cosmologies can be set by calling setCosmology(name). Note that changes in
# cosmological parameters are tracked to the fourth digit, which is why all parameters are rounded
# to at most four digits.
cosmologies = {}

# -------------------------------------------------------------------------------------------------
# Cosmologies from CMB data
# -------------------------------------------------------------------------------------------------

# planck1-only	Planck Collaboration 2013 	Best-fit, Planck only 					Table 2
# planck1		Planck Collaboration 2013 	Best-fit with BAO etc. 					Table 5
# WMAP9-only 	Hinshaw et al. 2013 		Max. likelihood, WMAP only 				Table 1
# WMAP9-ML 		Hinshaw et al. 2013 		Max. likelihood, with eCMB, BAO and H0 	Table 1
# WMAP9 		Hinshaw et al. 2013 		Best-fit, with eCMB, BAO and H0 		Table 4
# WMAP7-only 	Komatsu et al. 2011 		Max. likelihood, WMAP only 				Table 1
# WMAP7-ML 		Komatsu et al. 2011 		Max. likelihood, with BAO and H0 		Table 1
# WMAP7 		Komatsu et al. 2011 		Best-fit, with BAO and H0 				Table 1
# WMAP5-only 	Komatsu et al. 2009 		Max. likelihood, WMAP only 				Table 1
# WMAP5-ML 		Komatsu et al. 2009 		Max. likelihood, with BAO and SN 		Table 1
# WMAP5 		Komatsu et al. 2009 		Best-fit, with BAO and SN 				Table 1
# WMAP3-ML 		Spergel et al. 2007 		Max.likelihood, WMAP only 				Table 2
# WMAP3 		Spergel et al. 2007 		Best fit, WMAP only 					Table 5
# WMAP1-ML 		Spergel et al. 2005 		Max.likelihood, WMAP only 				Table 1 / 4
# WMAP1 		Spergel et al. 2005 		Best fit, WMAP only 					Table 7 / 4

cosmologies['planck1-only'] = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.0490, 'sigma8': 0.8344, 'ns': 0.9624}
cosmologies['planck1']      = {'flat': True, 'H0': 67.77, 'Om0': 0.3071, 'Ob0': 0.0483, 'sigma8': 0.8288, 'ns': 0.9611}
cosmologies['WMAP9-only']   = {'flat': True, 'H0': 69.70, 'Om0': 0.2814, 'Ob0': 0.0464, 'sigma8': 0.8200, 'ns': 0.9710}
cosmologies['WMAP9-ML']     = {'flat': True, 'H0': 69.70, 'Om0': 0.2821, 'Ob0': 0.0461, 'sigma8': 0.8170, 'ns': 0.9646}
cosmologies['WMAP9']        = {'flat': True, 'H0': 69.32, 'Om0': 0.2865, 'Ob0': 0.0463, 'sigma8': 0.8200, 'ns': 0.9608}
cosmologies['WMAP7-only']   = {'flat': True, 'H0': 70.30, 'Om0': 0.2711, 'Ob0': 0.0451, 'sigma8': 0.8090, 'ns': 0.9660}
cosmologies['WMAP7-ML']     = {'flat': True, 'H0': 70.40, 'Om0': 0.2715, 'Ob0': 0.0455, 'sigma8': 0.8100, 'ns': 0.9670}
cosmologies['WMAP7']        = {'flat': True, 'H0': 70.20, 'Om0': 0.2743, 'Ob0': 0.0458, 'sigma8': 0.8160, 'ns': 0.9680}
cosmologies['WMAP5-only']   = {'flat': True, 'H0': 72.40, 'Om0': 0.2495, 'Ob0': 0.0432, 'sigma8': 0.7870, 'ns': 0.9610}
cosmologies['WMAP5-ML']     = {'flat': True, 'H0': 70.20, 'Om0': 0.2769, 'Ob0': 0.0459, 'sigma8': 0.8170, 'ns': 0.9620}
cosmologies['WMAP5']        = {'flat': True, 'H0': 70.50, 'Om0': 0.2732, 'Ob0': 0.0456, 'sigma8': 0.8120, 'ns': 0.9600}
cosmologies['WMAP3-ML']     = {'flat': True, 'H0': 73.20, 'Om0': 0.2370, 'Ob0': 0.0414, 'sigma8': 0.7560, 'ns': 0.9540}
cosmologies['WMAP3']        = {'flat': True, 'H0': 73.50, 'Om0': 0.2342, 'Ob0': 0.0413, 'sigma8': 0.7420, 'ns': 0.9510}
cosmologies['WMAP1-ML']     = {'flat': True, 'H0': 68.00, 'Om0': 0.3136, 'Ob0': 0.0497, 'sigma8': 0.9000, 'ns': 0.9700}
cosmologies['WMAP1']        = {'flat': True, 'H0': 72.00, 'Om0': 0.2700, 'Ob0': 0.0463, 'sigma8': 0.9000, 'ns': 0.9900}

# -------------------------------------------------------------------------------------------------
# Cosmologies used in major N-body simulations
# -------------------------------------------------------------------------------------------------

# bolshoi		Klypin et al. 2011			Cosmology of the Bolshoi simulation
# millennium	Springel et al. 2005		Cosmology of the Millennium simulation 

cosmologies['bolshoi']      = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 'ns': 0.9500}
cosmologies['millennium']   = {'flat': True, 'H0': 73.00, 'Om0': 0.2500, 'Ob0': 0.0450, 'sigma8': 0.9000, 'ns': 1.0000}

# -------------------------------------------------------------------------------------------------
# Non-standard cosmologies
# -------------------------------------------------------------------------------------------------

# A generic cosmology for power-law spectra, designed to have similar H0 and sigma8 as the Bolshoi
# simulation
cosmologies['powerlaw']     = {'flat': True, 'H0': 70.00, 'Om0': 1.0000, 'Ob0': 0.0000, 'sigma8': 0.8200, 'ns': 1.0000}

###################################################################################################
# Cosmology class
###################################################################################################

# Most parameters must be set by the user, possibly by using the setCosmology() function using one 
# of the dictionaries above. The Cosmology class has the following parameters:
#
# -------------------------------------------------------------------------------------------------
# Parameter		Default		Description
# -------------------------------------------------------------------------------------------------
# name			None		A name for the cosmology, e.g. 'planck'.
# flat			True		If flat, there is no curvature, Omega_k = 0, and Omega_Lambda is 
#							calculated directly from Omega_m.
# Om0			None		Omega_m at z = 0.
# OL0			None		Omega_Lambda at z = 0. This parameter is ignored if flat == True.
# Ob0			None		Omega_baryon at z = 0.
# H0			None		The Hubble constant in km / s / Mpc.
# sigma8		None		The normalization of the power spectrum, i.e. the variance when
#							the field is filtered with a top hat filter of radius 8 Mpc/h.
# ns			None		The tilt of the primordial power spectrum.
# Tcmb0			2.725		The temperature of the CMB today in Kelvin.
# power_law		False		Assume a power-law matter power spectrum, P(k) = k^power_law_n.
# power_law_n	0.0			See above.
# data_dir		'Data/'		Directory where persistent data is stored. This path is relative
#							to the location of this file, not the execution directory.
# interpolation True        By default, lookup tables are created for certain computationally 
#                           intensive quantities, cutting down the computation times for future
#                           calculations. If interpolation == False, all interpolation is switched
#                           off. This can be useful when evaluating quantities for many different
#                           cosmologies (where computing the tables takes a prohibitively long 
#                           time). However, many functions will be MUCH slower if this setting is
#                           False, please use it only if absolutely necessary. Furthermore, the 
#                           derivative functions of P(k), sigma(R) etc will not work if 
#                           interpolation == False.
# storage       True        By default, the interpolation tables and such are stored in a permanent
#                           file. This avoids re-computing the tables when the same cosmology is 
#                           called again. However, if any file access is to be avoided (for example
#                           in MCMC chains), the user can set storage = False.
# print_info    False       Output information to the console.
# print_warningsFalse       Output warnings to the console.
# text_output	False		If text_output == True, all persistent data (such as lookup tables for 
#							sigma(R), the growth factor etc) is written into named text files in
#							addition to the default storage system. This feature allows the use
#							of these tables outside of this module.
#							
#							WARNING: Be careful with the text_output feature. Changes in cosmology
#							are not necessarily reflected in the text file names, and they can thus
#							overwrite the correct values. Always remove the text files from the 
#							directory after use. 
# -------------------------------------------------------------------------------------------------

class Cosmology():
	
	def __init__(self, name = None, flat = True, \
		Om0 = None, OL0 = None, Ob0 = None, H0 = None, sigma8 = None, ns = None, Tcmb0 = 2.725, \
		power_law = False, power_law_n = 0.0, \
		print_info = False, print_warnings = True, \
		interpolation = True, storage = True, text_output = False):
		
		if name == None:
			raise Exception('A name for the cosmology must be set.')
		if Om0 == None:
			raise Exception('Parameter Om0 must be set.')
		if Ob0 == None:
			raise Exception('Parameter Ob0 must be set.')
		if H0 == None:
			raise Exception('Parameter H0 must be set.')
		if sigma8 == None:
			raise Exception('Parameter sigma8 must be set.')
		if ns == None:
			raise Exception('Parameter ns must be set.')
		if Tcmb0 == None:
			raise Exception('Parameter Tcmb0 must be set.')
		if power_law and power_law_n == None:
			raise Exception('For a power-law cosmology, power_law_n must be set.')
		if not flat and OL0 == None:
			raise Exception('OL0 must be set for non-flat cosmologies.')
	
		self.name = name
		self.flat = flat
		self.power_law = power_law
		self.power_law_n = power_law_n
		self.Om0 = Om0
		self.OL0 = OL0
		self.Ob0 = Ob0
		self.H0 = H0
		self.h = H0 / 100.0
		self.Omh2 = self.Om0 * self.h**2
		self.Ombh2 = self.Ob0 * self.h**2
		self.sigma8 = sigma8
		self.ns = ns
		self.Tcmb0 = Tcmb0
		
		# Make sure flatness is obeyed
		self.ensureConsistency()
		
		# Flag for interpolation tables, storage, printing etc
		self.interpolation = interpolation
		self.storage = storage
		self.text_output = text_output
		self.print_info = print_info
		self.print_warnings = print_warnings
		
		# Lookup table for the linear growth factor, D+(z).
		self.z_max_Dplus = 1000.0
		self.z_Nbins_Dplus = 40
		
		# Lookup table for P(k). The Pk_norm field is only needed if interpolation == False.
		# Note that the binning is highly irregular for P(k), since much more resolution is
		# needed at the BAO scale and around the bend in the power spectrum. Thus, the binning
		# is split into multiple regions with different resolutions.
		self.k_Pk = [1E-20, 1E-4, 5E-2, 1E0, 1E6, 1E20]
		self.k_Pk_Nbins = [10, 30, 60, 20, 10]
		
		# Lookup table for sigma. Note that the nominal accuracy to which the integral is 
		# evaluated should match with the accuracy of the interpolation which is set by Nbins.
		# Here, they are matched to be accurate to better than ~3E-3.
		self.R_min_sigma = 1E-12
		self.R_max_sigma = 1E3
		self.R_Nbins_sigma = 18.0
		self.accuracy_sigma = 3E-3
	
		# Lookup table for correlation function xi
		self.R_xi = [1E-3, 5E1, 5E2]
		self.R_xi_Nbins = [30, 40]
		self.accuracy_xi = 1E-5
		
		# Data directory and storage dictionary. Note that the storage is active even if 
		# interpolation == False or storage == False, since a few numbers still need to be 
		# stored non-persistently.
		self.data_dir = 'cosmology'
		self.resetStorage()
		
		return

	###############################################################################################

	# Depending on whether the cosmology is flat or not, OL0 and Ok0 take on certain values.

	def ensureConsistency(self):
		
		if self.flat:
			self.OL0 = 1.0 - self.Om0
			self.Ok0 = 0.0
		else:
			self.Ok0 = 1.0 - self.OL0 - self.Om0

		return

	###############################################################################################

	# Check for changes in the cosmological parameters. If there are changes, various steps must
	# be taken to ensure that we are not outputting outdated values.

	def checkForChangedCosmology(self):
		
		hash_new = self.getHash()
		if hash_new != self.hash_current:
			if self.print_warnings:
				print("Cosmology: Detected change in cosmological parameters.")
			self.ensureConsistency()
			self.resetStorage()
			
		return
	
	###############################################################################################
	# Utilities for internal use
	###############################################################################################
	
	# Some functions permanently store lookup tables for faster execution. This function ensures
	# that this directory exists, and returns its path. The data directory is always created in the
	# directory where this file is located, rather than the execution directory which might vary, 
	# leading to unneccessary re-computations.
	
	def dataDir(self):
		
		path = Utilities.getCacheDir() + '/' + self.data_dir + '/'
		
		if not os.path.exists(path):
			os.makedirs(path)

		return path

	###############################################################################################

	# Compute a unique hash for the current cosmology name and parameters. If any of them change,
	# the hash will change, causing an update of stored quantities.
		
	def getHash(self):
	
		param_string = "Name_%s_Flat_%s_Om0_%.4f_OL0_%.4f_Ob0_%.4f_H0_%.4f_sigma8_%.4f_ns_%.4f_Tcmb0_%.4f_PL_%s_PLn_%.4f" \
			% (self.name, str(self.flat), self.Om0, self.OL0, self.Ob0, self.H0, self.sigma8, \
			self.ns, self.Tcmb0, str(self.power_law), self.power_law_n)

		hash_new = hashlib.md5(param_string).hexdigest()
	
		return hash_new
	
	###############################################################################################

	# Create a file name that is unique to this cosmology. While the hash encodes all necessary
	# information, the cosmology name is added to make it easier to identify the files with a 
	# cosmology.

	def getUniqueFilename(self):
		
		return self.dataDir() + self.name + '_' + self.getHash()
	
	###############################################################################################

	# Load stored objects. This function is called during the __init__() routine, and if a change
	# in cosmological parameters is detected.

	def resetStorage(self):

		# Reset the test hash and storage containers. There are two containes, one for objects
		# that are stored in a pickle file, and one for those that will be discarded when the 
		# class is destroyed.
		self.hash_current = self.getHash()
		self.storage_pers = {}
		self.storage_temp = {}
		
		# Check if there is a persistent object storage file. If so, load its contents into the
		# storage dictionary. We only load from file if the user has not switched of storage, and
		# if the user has not switched off interpolation.
		if self.storage and self.interpolation:
			filename_pickle = self.getUniqueFilename()
			if os.path.exists(filename_pickle):
				input_file = open(filename_pickle, "rb")
				self.storage_pers = pickle.load(input_file)
				input_file.close()

		return
	
	###############################################################################################

	# Permanent storage system for objects such as 2-dimensional data tables. If an object is 
	# already stored in memory, return it. If not, try to load it from file, otherwise return None.
	# Certain operations can already be performed on certain objects, so that they do not need to 
	# be repeated unnecessarily, for example:
	#
	# interpolator = True	Instead of a 2-dimensional table, return a spline interpolator that can
	#                       be used to evaluate the table.
	# inverse = True        Return an interpolator that gives x(y) instead of y(x)
	
	def getStoredObject(self, object_name, interpolator = False, inverse = False):
		
		# Check for cosmology change
		self.checkForChangedCosmology()

		# Compute object name
		object_id = object_name
		if interpolator:
			object_id += '_interpolator'
		if inverse:
			object_id += '_inverse'

		# Find the object. There are multiple possibilities:
		# - Check for the exact object the user requested (the object_id)
		#   - Check in persistent storage
		#   - Check in temporary storage (where interpolator / inverse objects live)
		#   - Check in user text files
		# - Check for the raw object (the object_name)
		#   - Check in persistent storage
		#   - Check in user text files
		#   - Convert to the exact object, store in temporary storage
		# - If all fail, return None

		if object_id in self.storage_pers:	
			object_data = self.storage_pers[object_id]
		
		elif object_id in self.storage_temp:	
			object_data = self.storage_temp[object_id]

		elif os.path.exists(self.dataDir() + object_id):
			object_data = numpy.loadtxt(self.dataDir() + object_id, usecols = (0, 1), \
									skiprows = 0, unpack = True)
			self.storage_temp[object_id] = object_data
			
		else:

			# We could not find the object ID anywhere. This can have two reasons: the object does
			# not exist, or we must transform an existing object.
			
			if interpolator:
				
				# First, a safety check; no interpolation objects should ever be requested if 
				# the user has switched off interpolation.
				if not self.interpolation:
					raise Exception('An interpolator object was requested even though interpolation is off.')
				
				# Try to find the object to transform. This object CANNOT be in temporary storage,
				# but it can be in persistent or user storage.
				object_raw = None
				
				if object_name in self.storage_pers:	
					object_raw = self.storage_pers[object_name]
		
				elif os.path.exists(self.dataDir() + object_name):
					object_raw = numpy.loadtxt(self.dataDir() + object_name, usecols = (0, 1), \
									skiprows = 0, unpack = True)

				if object_raw == None:
					
					# We cannot find an object to convert, return none.
					object_data = None
				
				else:
					
					# Convert and store in temporary storage.
					if inverse: 
						
						# There is a subtlety: the spline interpolator can't deal with decreasing 
						# x-values, so if the y-values 
						if object_raw[1][-1] < object_raw[1][0]:
							object_raw = object_raw[:,::-1]
						
						object_data = scipy.interpolate.InterpolatedUnivariateSpline(object_raw[1], \
																					object_raw[0])
					else:
						object_data = scipy.interpolate.InterpolatedUnivariateSpline(object_raw[0], \
																					object_raw[1])
					self.storage_temp[object_id] = object_data
						
			else:
							
				# The object is not in storage at all, and cannot be generated; return none.
				object_data = None
				
		return object_data
	
	###############################################################################################

	# Save an object in memory and file storage. If persistent == True, this object is written to 
	# file storage (unless storage == False), and will be loaded the next time the same cosmology
	# is loaded. If persistent == False, the object is stored non-persistently.
	#
	# Note that all objects are reset if the cosmology changes. Thus, this function should be used
	# for ALL data that depend on cosmological parameters.
	
	def storeObject(self, object_name, object_data, persistent = True):

		if persistent:
			self.storage_pers[object_name] = object_data
			
			if self.storage:
				# If the user has chosen text output, write a text file.
				if self.text_output:
					filename_text =  self.dataDir() + object_name
					numpy.savetxt(filename_text, numpy.transpose(object_data), fmt = "%.8e")
			
				# Store in file. We do not wish to save the entire storage dictionary, as there might be
				# user-defined objects in it.
				filename_pickle = self.getUniqueFilename()
				output_file = open(filename_pickle, "wb")
				pickle.dump(self.storage_pers, output_file, pickle.HIGHEST_PROTOCOL)
				output_file.close()  

		else:
			self.storage_temp[object_name] = object_data

		return
	
	###############################################################################################
	# Basic cosmology calculations
	###############################################################################################
	
	# The Hubble parameter as a function of redshift, in units of H0.
	
	def Ez(self, z):
		
		if self.flat:
			E = numpy.sqrt(self.Om0 * (1.0 + z)**3 + self.OL0)
		else:
			E = numpy.sqrt(self.Om0 * (1.0 + z)**3 + self.OL0 + self.Ok0 * (1.0 + z)**2)
			
		return E

	###############################################################################################
	
	# The Hubble parameter as a function of redshift, in units of km / s / Mpc.
	
	def Hz(self, z):
			
		return self.Ez(z) * self.H0

	###############################################################################################

	# Standard cosmological integrals. These integrals are not persistently stored in files because
	# they can be evaluated between any two redshifts which would make the tables very large.
	#
	# z_min and z_max can be numpy arrays or numbers. If one of the two is a number and the other an
	# array, the same z_min / z_max is used for all z_min / z_max in the array (this is useful if 
	# z_max = inf, for example).
	
	def integral(self, integrand, z_min, z_max):

		min_is_array = Utilities.isArray(z_min)
		max_is_array = Utilities.isArray(z_max)
		use_array = min_is_array or max_is_array
		
		if use_array and not min_is_array:
			z_min_use = numpy.array([z_min] * len(z_max))
		else:
			z_min_use = z_min
		
		if use_array and not max_is_array:
			z_max_use = numpy.array([z_max] * len(z_min))
		else:
			z_max_use = z_max
		
		if use_array:
			if min_is_array and max_is_array and len(z_min) != len(z_max):
				raise Exception("If both z_min and z_max are arrays, they need to have the same size.")
			integ = numpy.zeros((len(z_min_use)), numpy.float)
			for i in range(len(z_min_use)):
				integ[i], _ = scipy.integrate.quad(integrand, z_min_use[i], z_max_use[i])
		else:
			integ, _ = scipy.integrate.quad(integrand, z_min, z_max)
		
		return integ
	
	###############################################################################################

	# The integral over 1 / E(z) enters into the comoving distance.

	def integral_oneOverEz(self, z_min, z_max = numpy.inf):
		
		def integrand(z):
			return 1.0 / self.Ez(z)
		
		return self.integral(integrand, z_min, z_max)

	###############################################################################################

	# The integral over 1 / E(z) / (1 + z) enters into the age of the universe.

	def integral_oneOverEz1pz(self, z_min, z_max = numpy.inf):
		
		def integrand(z):
			return 1.0 / self.Ez(z) / (1.0 + z)
		
		return self.integral(integrand, z_min, z_max)

	###############################################################################################

	# The integral over (1 + z) / E(z)^3 enters into the linear growth factor.

	def integral_1pzOverEz3(self, z_min, z_max = numpy.inf):
		
		def integrand(z):
			return (1.0 + z) / (self.Ez(z))**3
		
		return self.integral(integrand, z_min, z_max)

	###############################################################################################
	# Times & distances
	###############################################################################################

	# 1 / H0 in units of Gyr.
	
	def hubbleTime(self):
		
		return 1E-16 * AST_Mpc / AST_year / self.h
	
	###############################################################################################

	# The lookback time between z = 0 and z in Gyr, i.e. the time difference between the age of the
	# universe at redshift z and today.

	def lookbackTime(self, z):
		
		return self.hubbleTime() * self.integral_oneOverEz1pz(0.0, z)
	
	###############################################################################################

	# The age of the universe at redshift z.
	
	def age(self, z = 0.0):
		
		return self.hubbleTime() * self.integral_oneOverEz1pz(z, numpy.inf)
	
	###############################################################################################

	# The comoving distance between redshift z_min and z_max in Mpc/h.
	
	def comovingDistance(self, z_min = 0.0, z_max = 0.0):
		
		return self.integral_oneOverEz(z_min = z_min, z_max = z_max) * AST_c * 1E-7

	###############################################################################################

	# The luminosity distance to redshift z in units of Mpc/h.

	def luminosityDistance(self, z):
		
		return self.comovingDistance(z_min = 0.0, z_max = z) * (1.0 + z)
	
	###############################################################################################

	# The angular diameter distance to redshift z in units of Mpc/h.

	def angularDiameterDistance(self, z):
		
		return self.comovingDistance(z_min = 0.0, z_max = z) / (1.0 + z)

	###############################################################################################

	# The distance modulus to redshift z in magnitudes.

	def distanceModulus(self, z):
		
		return 5.0 * numpy.log10(self.luminosityDistance(z) / self.h * 1E5)

	###############################################################################################

	# The sound horizon in Mpc (not Mpc / h!), according to Eisenstein & Hu 1998, equation 26. This 
	# fitting function is accurate to 2% where Obh2 > 0.0125 and 0.025 < Omh2 < 0.5.

	def soundHorizon(self):
		
		s = 44.5 * numpy.log(9.83 / self.Omh2) / numpy.sqrt(1.0 + 10.0 * self.Ombh2**0.75)
		
		return s

	###############################################################################################
	# Densities and overdensities
	###############################################################################################
	
	# The critical density in units of Msun h^2 / kpc^3.
	
	def criticalDensity(self, z):

		return AST_rho_crit_0_kpc3 * self.Ez(z)**2

	###############################################################################################

	# The matter density in units of Msun h^2 / kpc^3
	
	def matterDensity(self, z):

		return AST_rho_crit_0_kpc3 * self.Om0 * (1.0 + z)**3

	###############################################################################################

	# Omega_m(z)

	def Om(self, z):
			
		return self.Om0 * (1.0 + z)**3 / (self.Ez(z))**2

	###############################################################################################

	# Omega_Lambda(z)

	def OL(self, z):
			
		return self.OL0 / (self.Ez(z))**2

	###############################################################################################

	# Omega_k(z)

	def Ok(self, z):
			
		return self.Ok0 * (1.0 + z)**2 / (self.Ez(z))**2

	###############################################################################################
	# Structure growth, power spectrum etc.
	###############################################################################################

	# Convert the mass of a halo (in units of Msun / h) to the radius of ifs Lagrangian volume (in 
	# units or Mpc / h). This conversion is used when computing peak height, for example.
	
	def lagrangianR(self, M):
		
		return (3.0 * M / 4.0 / math.pi / self.matterDensity(0.0) / 1E9)**(1.0 / 3.0)
	
	###############################################################################################

	# Inverse of the function above, M (Msun / h) from R (Mpc / h).
	
	def lagrangianM(self, R):
		
		return 4.0 / 3.0 * math.pi * R**3 * self.matterDensity(0.0) * 1E9

	###############################################################################################

	# The linear growth factor, D+(z), as defined in Eisenstein & Hu 99, eq. 8. The normalization
	# is such that the growth factor approaches D+ -> 1/(1+z) at high z. There are other 
	# normalizations of this quantity (e.g., Percival 2005, eq. 15), but since we almost always 
	# care about the growth factor normalized to z = 0 the normalization does not matter.

	def growthFactorUnnormalized(self, z):
	
		return 5.0 / 2.0 * self.Om0 * self.Ez(z) * self.integral_1pzOverEz3(z)

	###############################################################################################

	# Return a spline interpolator for the growth factor. Generally, the growth factor should be 
	# evaluated using the growthFactor() function below.
	
	def growthFactorInterpolator(self):
		
		table_name = 'growthfactor_%s' % (self.name)
		interpolator = self.getStoredObject(table_name, interpolator = True)
		
		if interpolator == None:
			if self.print_info:
				print("Cosmology.growthFactor: Computing lookup table.")
			log_max = numpy.log10(1.0 + self.z_max_Dplus)
			bin_width = log_max / self.z_Nbins_Dplus
			z_table = 10**numpy.arange(0.0, log_max + bin_width, bin_width) - 1.0
			D_table = self.growthFactorUnnormalized(z_table) / self.growthFactorUnnormalized(0.0)
			table_ = numpy.array([z_table, D_table])
			self.storeObject(table_name, table_)
			if self.print_info:
				print("Cosmology.growthFactor: Lookup table completed.")
			interpolator = self.getStoredObject(table_name, interpolator = True)
		
		return interpolator

	###############################################################################################
	
	# The linear growth factor, normalized to z = 0.

	def growthFactor(self, z):
		
		if self.interpolation:
			interpolator = self.growthFactorInterpolator()
			if numpy.max(z) > self.z_max_Dplus:
				msg = "Cosmology.growthFactor: z = %.2f outside range (max. z is %.2f)." % (numpy.max(z), self.z_max_Dplus)
				raise Exception(msg)
			D = interpolator(z)
		
		else:
			D = self.growthFactorUnnormalized(z) / self.growthFactorUnnormalized(0.0)

		return D

	###############################################################################################
	
	# Return the critical overdensity for collapse. This can be approximated as a constant, or a 
	# correction for the ellipticity of peaks can be applied, according to Sheth et al. 2001.
	
	def collapseOverdensity(self, deltac_const = True, sigma = None):
		
		if deltac_const:
			delta_c = AST_delta_collapse
		else:
			delta_c = AST_delta_collapse * (1.0 + 0.47 * (sigma / AST_delta_collapse)**1.23)
		
		return delta_c
	
	###############################################################################################
	
	# The Eisenstein & Hu 1998 approximation to the transfer function at a scale k (h / Mpc), used 
	# below to compute the linear matter power spectrum. Based on Matt Becker's cosmocalc code.
	
	def transferFunctionEH98(self, k):
		
		# Define shorter expressions
		omb = self.Ob0
		om0 = self.Om0
		omc = om0 - omb
		h = self.h
		theta2p7 = self.Tcmb0 / 2.7
		
		# Convert kh from h/Mpc to 1/Mpc
		kh = k * h
	
		# Equation 2
		zeq = 2.50e4 * om0 * h * h / (theta2p7 * theta2p7 * theta2p7 * theta2p7)
	
		# Equation 3
		keq = 7.46e-2 * om0 * h * h / (theta2p7 * theta2p7)
	
		# Equation 4
		b1d = 0.313 * pow(om0 * h * h, -0.419) * (1.0 + 0.607 * pow(om0 * h * h, 0.674))
		b2d = 0.238 * pow(om0 * h * h, 0.223)
		zd = 1291.0 * pow(om0 * h * h, 0.251) / (1.0 + 0.659 * pow(om0 * h * h, 0.828)) \
			* (1.0 + b1d * pow(omb * h * h, b2d))
	
		# Equation 5
		Rd = 31.5 * omb * h * h / (theta2p7 * theta2p7 * theta2p7 * theta2p7) / ((zd) / 1e3)
		Req = 31.5 * omb * h * h / (theta2p7 * theta2p7 * theta2p7 * theta2p7) / (zeq / 1e3)
	
		# Equation 6
		s = 2.0 / 3.0 / keq * numpy.sqrt(6.0 / Req) * numpy.log((numpy.sqrt(1.0 + Rd) + \
			numpy.sqrt(Rd + Req)) / (1.0 + numpy.sqrt(Req)))
	
		# Equation 7
		ksilk = 1.6 * pow(omb * h * h, 0.52) * pow(om0 * h * h, 0.73) \
			* (1.0 + pow(10.4 * om0 * h * h, -0.95))
	
		# Equation 10
		q = kh / 13.41 / keq
	
		# Equation 11
		a1 = pow(46.9 * om0 * h * h, 0.670) * (1.0 + pow(32.1 * om0 * h * h, -0.532))
		a2 = pow(12.0 * om0 * h * h, 0.424) * (1.0 + pow(45.0 * om0 * h * h, -0.582))
		ac = pow(a1, -1.0 * omb / om0) * pow(a2, -1.0 * (omb / om0) * (omb / om0) * (omb / om0))
	
		# Equation 12
		b1 = 0.944 / (1.0 + pow(458.0 * om0 * h * h, -0.708))
		b2 = pow(0.395 * om0 * h * h, -0.0266)
		bc = 1.0 / (1.0 + b1 * (pow(omc / om0, b2) - 1.0))
	
		# Equation 15
		y = (1.0 + zeq) / (1.0 + zd)
		Gy = y * (-6.0 * numpy.sqrt(1.0 + y) + (2.0 + 3.0 * y) \
			* numpy.log((numpy.sqrt(1.0 + y) + 1.0) / (numpy.sqrt(1.0 + y) - 1.0)))
	
		# Equation 14
		ab = 2.07 * keq * s * pow(1.0 + Rd, -3.0 / 4.0) * Gy
	
		# Get CDM part of transfer function
	
		# Equation 18
		f = 1.0 / (1.0 + (kh * s / 5.4) * (kh * s / 5.4) * (kh * s / 5.4) * (kh * s / 5.4))
	
		# Equation 20
		C = 14.2 / ac + 386.0 / (1.0 + 69.9 * pow(q, 1.08))
	
		# Equation 19
		T0t = numpy.log(numpy.e + 1.8 * bc * q) / (numpy.log(numpy.e + 1.8 * bc * q) + C * q * q)
	
		# Equation 17
		C1bc = 14.2 + 386.0 / (1.0 + 69.9 * pow(q, 1.08))
		T0t1bc = numpy.log(numpy.e + 1.8 * bc * q) / (numpy.log(numpy.e + 1.8 * bc * q) + C1bc * q * q)
		Tc = f * T0t1bc + (1.0 - f) * T0t
	
		# Get baryon part of transfer function
	
		# Equation 24
		bb = 0.5 + omb / om0 + (3.0 - 2.0 * omb / om0) * numpy.sqrt((17.2 * om0 * h * h) \
			* (17.2 * om0 * h * h) + 1.0)
	
		# Equation 23
		bnode = 8.41 * pow(om0 * h * h, 0.435)
	
		# Equation 22
		st = s / pow(1.0 + (bnode / kh / s) * (bnode / kh / s) * (bnode / kh / s), 1.0 / 3.0)
	
		# Equation 21
		C11 = 14.2 + 386.0 / (1.0 + 69.9 * pow(q, 1.08))
		T0t11 = numpy.log(numpy.e + 1.8 * q) / (numpy.log(numpy.e + 1.8 * q) + C11 * q * q)
		Tb = (T0t11 / (1.0 + (kh * s / 5.2) * (kh * s / 5.2)) + ab / (1.0 + (bb / kh / s) * \
			(bb / kh / s) * (bb / kh / s)) * numpy.exp(-pow(kh / ksilk, 1.4))) \
			* numpy.sin(kh * st) / (kh * st)
	
		# Total transfer function
		Tk = omb / om0 * Tb + omc / om0 * Tc
	
		return Tk

	###############################################################################################
	
	# The Eisenstein & Hu 1998 transfer function at a scale k (h / Mpc) but without the BAO wiggles.
	
	def transferFunctionEH98Smooth(self, k):

		omb = self.Ob0
		om0 = self.Om0
		h = self.h
		theta2p7 = self.Tcmb0 / 2.7

		# Convert kh from hMpc^-1 to Mpc^-1
		kh = k * h
	
		# Equation 26
		s = 44.5 * numpy.log(9.83 / om0 / h / h) / numpy.sqrt(1.0 + 10.0 * pow(omb * h * h, 0.75))
	
		# Equation 31
		alphaGamma = 1.0 - 0.328 * numpy.log(431.0 * om0 * h * h) * omb / om0 \
				+ 0.38 * numpy.log(22.3 * om0 * h * h) * (omb / om0) * (omb / om0)
	
		# Equation 30
		Gamma = om0 * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + pow(0.43 * kh * s, 4.0)))
	
		# Equation 28
		q = k * theta2p7 * theta2p7 / Gamma
	
		# Equation 29
		C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
		L0 = numpy.log(2.0 * numpy.exp(1.0) + 1.8 * q)
		Tk = L0 / (L0 + C0 * q * q)
	
		return Tk	

	###############################################################################################
	
	# Evaluate the matter power spectrum. If ignore_norm == True, do not normalize by sigma8; this
	# mode is for internal use only. k can be a number, or a numpy array.
	
	def matterPowerSpectrumExact(self, k, Pk_source = 'eh98', ignore_norm = False):
		
		if self.power_law:
			
			Pk_source = 'powerlaw'
			Pk = k**self.power_law_n
			
		elif Pk_source == 'eh98':

			T = self.transferFunctionEH98(k)
			Pk = T * T * k**self.ns

		elif Pk_source == 'eh98smooth':

			T = self.transferFunctionEH98Smooth(k)
			Pk = T * T * k**self.ns

		else:
			
			table_name = 'matterpower_%s_%s' % (self.name, Pk_source)
			table = self.getStoredObject(table_name)

			if table == None:
				msg = "Could not load data table, %s." % (table_name)
				raise Exception(msg)
			if numpy.max(k) > numpy.max(table[0]):
				msg = "k (%.2e) is larger than max. k (%.2e)." % (numpy.max(k), numpy.max(table[0]))
				raise Exception(msg)
			if numpy.min(k) < numpy.min(table[0]):
				msg = "k (%.2e) is smaller than min. k (%.2e)." % (numpy.min(k), numpy.min(table[0]))
				raise Exception(msg)

			Pk = numpy.interp(k, table[0], table[1])
		
		# This is a little tricky. We need to store the normalization factor somewhere, even if 
		# interpolation = False; otherwise, we get into an infinite loop of computing sigma8, P(k), 
		# sigma8 etc.
		if not ignore_norm:
			norm_name = 'Pk_norm_%s_%s' % (self.name, Pk_source)
			norm = self.getStoredObject(norm_name)
			if norm == None:
				sigma_8Mpc = self.sigmaExact(8.0, filt = 'tophat', Pk_source = Pk_source, \
											exact_Pk = True, ignore_norm = True)
				norm = (self.sigma8 / sigma_8Mpc)**2
				self.storeObject(norm_name, norm, persistent = False)

			Pk *= norm
	
		return Pk
	
	###############################################################################################

	# Utility to get the min and max k for which a power spectrum is valid. Only for internal use.

	def matterPowerSpectrumLimits(self, Pk_source):
		
		if self.power_law or Pk_source == 'eh98' or Pk_source == 'eh98smooth':
			k_min = self.k_Pk[0]
			k_max = self.k_Pk[-1]
		else:
			table_name = 'matterpower_%s_%s' % (self.name, Pk_source)
			table = self.getStoredObject(table_name)

			if table == None:
				msg = "Could not load data table, %s." % (table_name)
				raise Exception(msg)
	
			k_min = table[0][0]
			k_max = table[0][-1]
		
		return k_min, k_max
	
	###############################################################################################

	# Return a spline interpolator for the power spectrum. Generally, P(k) should be evaluated 
	# using the matterPowerSpectrum() function below, but for some performance-critical operations
	# it is faster to obtain the interpolator directly from this function. Note that the lookup 
	# table created here is complicated, with extra resolution around the BAO scale.

	def matterPowerSpectrumInterpolator(self, Pk_source):
		
		table_name = 'Pk_%s_%s' % (self.name, Pk_source)
		interpolator = self.getStoredObject(table_name, interpolator = True)
	
		if interpolator == None:
			if self.print_info:
				print("Cosmology.matterPowerSpectrum: Computing lookup table.")				
			data_k = numpy.zeros((numpy.sum(self.k_Pk_Nbins) + 1), numpy.float)
			n_regions = len(self.k_Pk_Nbins)
			k_computed = 0
			for i in range(n_regions):
				log_min = numpy.log10(self.k_Pk[i])
				log_max = numpy.log10(self.k_Pk[i + 1])
				log_range = log_max - log_min
				bin_width = log_range / self.k_Pk_Nbins[i]
				if i == n_regions - 1:
					data_k[k_computed:k_computed + self.k_Pk_Nbins[i] + 1] = \
						10**numpy.arange(log_min, log_max + bin_width, bin_width)
				else:
					data_k[k_computed:k_computed + self.k_Pk_Nbins[i]] = \
						10**numpy.arange(log_min, log_max, bin_width)
				k_computed += self.k_Pk_Nbins[i]
			
			data_Pk = self.matterPowerSpectrumExact(data_k, Pk_source = Pk_source, ignore_norm = False)
			table_ = numpy.array([numpy.log10(data_k), numpy.log10(data_Pk)])
			self.storeObject(table_name, table_)
			if self.print_info:
				print("Cosmology.matterPowerSpectrum: Lookup table completed.")	
			
			interpolator = self.getStoredObject(table_name, interpolator = True)

		return interpolator

	###############################################################################################

	# Return the power spectrum at a scale k (h / Mpc), either computed by Eisenstein & Hu 98 
	# (Pk_source = 'eh98', 'eh98smooth') or another method (e.g. Pk_source = 'camb'). If the latter 
	# is chosen, the user must place a file with the power spectrum data (k, P(k)) in the data 
	# directory, and name it matterpower_<cosmo_name>_<Pk_source>, e.g. matterpower_planck_camb. 
	#
	# For the EH98 P(k) sources, this function creates a lookup table with generous limits in k
	# space. If, for some reason, the exact value is desired, use the matterPowerSpectrumExact()
	# function. The EH98 power spectrum is accurate to about 1%.
	#
	# If derivative == True, the logarithmic derivative is returned, d log(P) / d log(k).

	def matterPowerSpectrum(self, k, Pk_source = 'eh98', derivative = False):
		
		if self.interpolation and (Pk_source == 'eh98' or Pk_source == 'eh98smooth'):
			
			# Load lookup-table
			interpolator = self.matterPowerSpectrumInterpolator(Pk_source)
			
			# If the requested radius is outside the range, give a detailed error message.
			k_req = numpy.min(k)
			if k_req < self.k_Pk[0]:
				msg = "k = %.2e is too small (min. k = %.2e)" % (k_req, self.k_min_Pk)
				raise Exception(msg)
		
			k_req = numpy.max(k)
			if k_req > self.k_Pk[-1]:
				msg = "k = %.2e is too large (max. k = %.2e)" % (k_req, self.k_max_Pk)
				raise Exception(msg)

			if derivative:
				Pk = interpolator(numpy.log10(k), nu = 1)
			else:
				Pk = interpolator(numpy.log10(k))
				Pk = 10**Pk
			
		else:
			
			if derivative > 0:
				raise Exception("Derivative can only be evaluated if interpolation == True.")

			if Utilities.isArray(k):
				Pk = k * 0.0
				for i in range(len(k)):
					Pk[i] = self.matterPowerSpectrumExact(k[i], Pk_source = Pk_source, ignore_norm = False)
			else:
				Pk = self.matterPowerSpectrumExact(k, Pk_source = Pk_source, ignore_norm = False)

		return Pk
	
	###############################################################################################

	# The k-space filter function. This function is dimensionless, the input units are k in h / Mpc
	# and R in Mpc / h. Possible filters are 'tophat' and 'gaussian'. If no_oscillation == True, we
	# are using the filter for an estimate.

	def filterFunction(self, filt, k, R, no_oscillation = False):
		
		x = k * R
		
		if filt == 'tophat':
			
			if no_oscillation:
				if x < 1.0:
					ret = 1.0
				else:
					ret = x**-2
			else:
				if x < 1E-3:
					ret = 1.0
				else:
					ret = 3.0 / x**3 * (numpy.sin(x) - x * numpy.cos(x))
				
		elif filt == 'gaussian':
			ret = numpy.exp(-x**2)
		
		else:
			msg = "Invalid filter, %s." % (filt)
			raise Exception(msg)
			
		return ret

	###############################################################################################

	# See documentation of sigma() function below. This function performs the actual calculation of
	# sigma, and can be called directly by the user. However, this function will take much longer
	# than the lookup table below. If exact_Pk == True and ignore_norm == True, the unnormalized 
	# power spectrum is used. This mode should only be used internally to compute the normalization 
	# in the first place.
	#
	# NOTE: This function does NOT accept a numpy integral for R, only a number.
	#
	# Accuracy: This function computes the integral to the accuracy set in self.accuracy_sigma, by
	#           default 0.3%. Note that the power spectrum is generally known less accurately than 
	#           that, for example if the EH98 approximation is used.

	def sigmaExact(self, R, j = 0, filt = 'tophat', Pk_source = 'eh98', \
				exact_Pk = False, ignore_norm = False):
		
		# -----------------------------------------------------------------------------------------
		def logIntegrand(lnk, Pk_interpolator, test = False):
			
			k = numpy.exp(lnk)
			W = self.filterFunction(filt, k, R, no_oscillation = test)
			
			if exact_Pk or (not self.interpolation):
				Pk = self.matterPowerSpectrumExact(k, Pk_source = Pk_source, ignore_norm = ignore_norm)
			else:
				Pk = 10**Pk_interpolator(numpy.log10(k))
			
			# One factor of k is due to the integration in log-k space
			ret = Pk * W**2 * k**3
			
			# Higher moment terms
			if j > 0:
				ret *= k**(2 * j)
			
			return ret

		# -----------------------------------------------------------------------------------------
		if filt == 'tophat' and j > 0:
			msg = "Higher-order moments of sigma are not well-defined for " + "tophat filter. Choose filt = 'gaussian' instead."
			raise Exception(msg)
	
		# For power-law cosmologies, we can evaluate sigma analytically. The exact expression 
		# has a dependence on n that in turn depends on the filter used, but the dependence 
		# on radius is simple and independent of the filter. Thus, we use sigma8 to normalize
		# sigma directly. 
		if self.power_law:
			
			n = self.power_law_n + 2 * j
			if n <= -3.0:
				msg = "n + 2j must be > -3 for the variance to converge in a power-law cosmology."
				raise Exception(msg)
			sigma2 = R**(-3 - n) / (8.0**(-3 - n) / self.sigma8**2)
			sigma = numpy.sqrt(sigma2)
			
		else:
			
			# If we are getting P(k) from a look-up table, it is a little more efficient to 
			# get the interpolator object and use it directly, rather than using the P(k) function.
			Pk_interpolator = None
			if (not exact_Pk) and self.interpolation:
				Pk_interpolator = self.matterPowerSpectrumInterpolator(Pk_source)
			
			# The infinite integral over k often causes trouble when the tophat filter is used. Thus,
			# we determine sensible limits and integrate over a finite volume. For tabled power 
			# spectra, we need to be careful not to exceed their limits.
			test_integrand_min = 1E-6
			test_k_min, test_k_max = self.matterPowerSpectrumLimits(Pk_source)
			test_k_min = max(test_k_min * 1.0001, 1E-7)
			test_k_max = min(test_k_max * 0.9999, 1E15)
			test_k = numpy.arange(numpy.log(test_k_min), numpy.log(test_k_max), 2.0)
			n_test = len(test_k)
			test_k_integrand = test_k * 0.0
			for i in range(n_test):
				test_k_integrand[i] = logIntegrand(test_k[i], Pk_interpolator)
			integrand_max = numpy.max(test_k_integrand)
			
			min_index = 0
			while test_k_integrand[min_index] < integrand_max * test_integrand_min:
				min_index += 1
				if min_index > n_test - 2:
					msg = "Could not find lower integration limit."
					raise Exception(msg)

			min_index -= 1
			max_index = min_index + 1
			while test_k_integrand[max_index] > integrand_max * test_integrand_min:
				max_index += 1	
				if max_index == n_test:
					msg = "Could not find upper integration limit."
					raise Exception(msg)
	
			args = Pk_interpolator
			sigma2, _ = scipy.integrate.quad(logIntegrand, test_k[min_index], test_k[max_index], \
						args = args, epsabs = 0.0, epsrel = self.accuracy_sigma, limit = 100)
			sigma = numpy.sqrt(sigma2 / 2.0 / math.pi**2)
		
		if numpy.isnan(sigma):
			msg = "Result is nan (cosmology %s, filter %s, R %.2e, j %d." % (self.name, filt, R, j)
			raise Exception(msg)
			
		return sigma
	
	###############################################################################################

	# Return a spline interpolator for sigma(R) or R(sigma) if inverse == True. Generally, sigma(R) 
	# should be evaluated using the sigma() function below, but for some performance-critical 
	# operations it is faster to obtain the interpolator directly from this function.If the lookup-
	# table does not exist yet, create it. For sigma, we use a very particular binning scheme. At 
	# low R, sigma is a very smooth function, and very wellapproximated by a spline interpolation 
	# between few points. Around the BAO scale, we need a higher resolution. Thus, the bins are 
	# assigned in reverse log(log) space.

	def sigmaInterpolator(self, j, Pk_source, filt, inverse):
		
		table_name = 'sigma%d_%s_%s_%s' % (j, self.name, Pk_source, filt)
		interpolator = self.getStoredObject(table_name, interpolator = True, inverse = inverse)
		
		if interpolator == None:
			if self.print_info:
				print("Cosmology.sigma: Computing lookup table.")
			max_log = numpy.log10(self.R_max_sigma)
			log_range = max_log - numpy.log10(self.R_min_sigma)
			max_loglog = numpy.log10(log_range + 1.0)
			loglog_width = max_loglog / self.R_Nbins_sigma
			R_loglog = numpy.arange(0.0, max_loglog + loglog_width, loglog_width)
			log_R = max_log - 10**R_loglog[::-1] + 1.0
			data_R = 10**log_R
			data_sigma = data_R * 0.0
			for i in range(len(data_R)):
				data_sigma[i] = self.sigmaExact(data_R[i], j = j, filt = filt, Pk_source = Pk_source)
			table_ = numpy.array([numpy.log10(data_R), numpy.log10(data_sigma)])
			self.storeObject(table_name, table_)
			if self.print_info:
				print("Cosmology.sigma: Lookup table completed.")

			interpolator = self.getStoredObject(table_name, interpolator = True, inverse = inverse)
	
		return interpolator

	###############################################################################################

	# The variance of the linear density field, sigma, smoothed over a scale R (Mpc / h) and 
	# normalized such that sigma(8 Mpc/h) = sigma8. This function is accurate to ~1% if the EH98
	# power spectrum is used.
	#
	# ---------------------------------------------------------------------------------------------
	# Parameter		Default		Description
	# ---------------------------------------------------------------------------------------------
	# R				None		The radius of the filter, in Mpc / h
	# j				0			The order of the integral. j = 0 corresponds to the variance, j = 1
	#							to the same integral with an extra k^2 term etc. See BBKS for the 
	#							mathematical details.
	# z				0.0			Redshift; for z != 0, sigma is multiplied by the growth factor.
	# filt			'tophat'	The filter used; 'tophat' by default, but 'gaussian' can also be 
	#							chosen.
	# Pk_source		'eh98'		The source of the underlying power spectrum (eh98, eh98smooth or 
	#							tabulated). For power-law cosmologies, this parameter is ignored.
	# inverse		False		Compute R(sigma) rather than sigma(R), using the same lookup table.
	# derivative	False		Return the logarithmic derivative, d log(sigma) / d log(R), or its
	#                           inverse, d log(R) / d log(sigma), if inverse == True.
	# ---------------------------------------------------------------------------------------------
	
	def sigma(self, R, j = 0, z = 0.0, inverse = False, derivative = False, \
			Pk_source = 'eh98', filt = 'tophat'):

		if self.interpolation:
			interpolator = self.sigmaInterpolator(j, Pk_source, filt, inverse)
			
			if not inverse:
	
				# If the requested radius is outside the range, give a detailed error message.
				R_req = numpy.min(R)
				if R_req < self.R_min_sigma:
					M_min = 4.0 / 3.0 * math.pi * self.R_min_sigma**3 * self.matterDensity(0.0) * 1E9
					msg = "R = %.2e is too small (min. R = %.2e, min. M = %.2e)" \
						% (R_req, self.R_min_sigma, M_min)
					raise Exception(msg)
			
				R_req = numpy.max(R)
				if R_req > self.R_max_sigma:
					M_max = 4.0 / 3.0 * math.pi * self.R_max_sigma**3 * self.matterDensity(0.0) * 1E9
					msg = "R = %.2e is too large (max. R = %.2e, max. M = %.2e)" \
						% (R_req, self.R_max_sigma, M_max)
					raise Exception(msg)
	
				if derivative:
					ret = interpolator(numpy.log10(R), nu = 1)
				else:
					ret = 10**interpolator(numpy.log10(R))
					if z > 1E-5:
						ret *= self.growthFactor(z)
	
			else:
				
				sigma_ = R
				if z > 1E-5:
					sigma_ /= self.growthFactor(z)

				# Get the limits in sigma from storage, or compute and store them. Using the 
				# storage mechanism seems like overkill, but these numbers should be erased if 
				# the cosmology changes and sigma is re-computed.
				sigma_min = self.getStoredObject('sigma_min')
				sigma_max = self.getStoredObject('sigma_max')
				if sigma_min == None or sigma_min == None:
					knots = interpolator.get_knots()
					sigma_min = 10**numpy.min(knots)
					sigma_max = 10**numpy.max(knots)
					self.storeObject('sigma_min', sigma_min, persistent = False)
					self.storeObject('sigma_max', sigma_max, persistent = False)
				
				# If the requested sigma is outside the range, give a detailed error message.
				sigma_req = numpy.max(sigma_)
				if sigma_req > sigma_max:
					msg = "sigma = %.2e is too large (max. sigma = %.2e)" % (sigma_req, sigma_max)
					raise Exception(msg)
					
				sigma_req = numpy.min(sigma_)
				if sigma_req < sigma_min:
					msg = "sigma = %.2e is too small (min. sigma = %.2e)" % (sigma_req, sigma_min)
					raise Exception(msg)
				
				# Interpolate to get R(sigma)
				if derivative: 
					ret = interpolator(numpy.log10(sigma_), nu = 1)					
				else:
					ret = 10**interpolator(numpy.log10(sigma_))
		
		else:
			
			if inverse:
				raise Exception('R(sigma), and thus nu_to_M(), cannot be evaluated with interpolation == False.')
			if derivative:
				raise Exception('Derivative of sigma cannot be evaluated if interpolation == False.')

			if Utilities.isArray(R):
				ret = R * 0.0
				for i in range(len(R)):
					ret[i] = self.sigmaExact(R[i], j = j, filt = filt, Pk_source = Pk_source)
			else:
				ret = self.sigmaExact(R, j = j, filt = filt, Pk_source = Pk_source)
			if z > 1E-5:
				ret *= self.growthFactor(z)
		
		return ret

	###############################################################################################
	
	# Computes peak height, nu, from mass in units of Msun / h. See the documentation of the sigma 
	# function above for possible filters, Pk sources etc.
	
	def M_to_nu(self, M, z, filt = 'tophat', Pk_source = 'eh98', deltac_const = True):
		
		R = self.lagrangianR(M)
		sigma = self.sigma(R, z = z, filt = filt, Pk_source = Pk_source)
		nu = self.collapseOverdensity(deltac_const, sigma) / sigma

		return nu
	
	###############################################################################################
	
	# The inverse of the function above.
	
	def nu_to_M(self, nu, z, filt = 'tophat', Pk_source = 'eh98', deltac_const = True):

		sigma = self.collapseOverdensity(deltac_const = deltac_const) / nu
		R = self.sigma(sigma, z = z, filt = filt, Pk_source = Pk_source, inverse = True)
		M = self.lagrangianM(R)
		
		return M
	
	###############################################################################################
	
	# The non-linear mass M* (or M_NL), i.e. the mass for which the variance is equal to the 
	# collapse threshold.
	
	def nonLinearMass(self, z, filt = 'tophat', Pk_source = 'eh98'):
		
		return self.nu_to_M(1.0, z = z, filt = filt, Pk_source = Pk_source, deltac_const = True)

	###############################################################################################

	# Compute the linear matter-matter correlation function at scale R in Mpc / h at z = 0. This 
	# function does NOT accept a numpy integral for R, only a number. This function is accurate to 
	# ~1-2% in for 1E-3 < R < 500. 

	def correlationFunctionExact(self, R, Pk_source = 'eh98'):

		f_cut = 0.001

		# -----------------------------------------------------------------------------------------
		# The integrand is exponentially cut off at a scale 1000 * R.
		def integrand(k, R, Pk_source, Pk_interpolator):
			
			if self.interpolation:
				Pk = 10**Pk_interpolator(numpy.log10(k))
			else:
				Pk = self.matterPowerSpectrumExact(k, Pk_source)

			ret = Pk * k / R * numpy.exp(-(k * R * f_cut)**2)
			
			return ret

		# -----------------------------------------------------------------------------------------
		# If we are getting P(k) from a look-up table, it is a little more efficient to 
		# get the interpolator object and use it directly, rather than using the P(k) function.
		Pk_interpolator = None
		if self.interpolation:
			Pk_interpolator = self.matterPowerSpectrumInterpolator(Pk_source)

		# Use a Clenshaw-Curtis integration, i.e. an integral weighted by sin(kR). 
		k_min = 1E-6 / R
		k_max = 10.0 / f_cut / R
		args = R, Pk_source, Pk_interpolator
		xi, _ = scipy.integrate.quad(integrand, k_min, k_max, args = args, epsabs = 0.0, \
					epsrel = self.accuracy_xi, limit = 100, weight = 'sin', wvar = R)
		xi /= 2.0 * math.pi**2

		if numpy.isnan(xi):
			msg = 'Result is nan (cosmology %s, R %.2e).' % (self.name, R)
			raise Exception(msg)

		return xi
	
	###############################################################################################

	# Return a spline interpolator for the correlation function, xi(R). Generally, xi(R) should be 
	# evaluated using the correlationFunction() function below, but for some performance-critical 
	# operations it is faster to obtain the interpolator directly from this function.

	def correlationFunctionInterpolator(self, Pk_source):

		table_name = 'correlation_%s_%s' % (self.name, Pk_source)
		interpolator = self.getStoredObject(table_name, interpolator = True)
		
		if interpolator == None:
			if self.print_info:
				print("correlationFunction: Computing lookup table. This may take a few minutes, please do not interrupt.")
			
			data_R = numpy.zeros((numpy.sum(self.R_xi_Nbins) + 1), numpy.float)
			n_regions = len(self.R_xi_Nbins)
			k_computed = 0
			for i in range(n_regions):
				log_min = numpy.log10(self.R_xi[i])
				log_max = numpy.log10(self.R_xi[i + 1])
				log_range = log_max - log_min
				bin_width = log_range / self.R_xi_Nbins[i]
				if i == n_regions - 1:
					data_R[k_computed:k_computed + self.R_xi_Nbins[i] + 1] = \
						10**numpy.arange(log_min, log_max + bin_width, bin_width)
				else:
					data_R[k_computed:k_computed + self.R_xi_Nbins[i]] = \
						10**numpy.arange(log_min, log_max, bin_width)
				k_computed += self.R_xi_Nbins[i]
			
			data_xi = data_R * 0.0
			for i in range(len(data_R)):
				data_xi[i] = self.correlationFunctionExact(data_R[i], Pk_source = Pk_source)
			table_ = numpy.array([data_R, data_xi])
			self.storeObject(table_name, table_)
			if self.print_info:
				print("correlationFunction: Lookup table completed.")
			interpolator = self.getStoredObject(table_name, interpolator = True)
		
		return interpolator

	###############################################################################################

	# The linear matter-matter correlation function as function of radius in Mpc / h. This function
	# as well as the interpolation routine are accurate to ~1-2%. Note that evaluating this 
	# function is relatively expensive due to the nature of the xi integral. If you only need to 
	# evaluate xi at a few R, setting interpolation = False might speed up the computation.
	#
	# If derivative == True, the linear derivative d xi / d R is returned.

	def correlationFunction(self, R, z = 0.0, derivative = False, Pk_source = 'eh98'):

		if self.interpolation:
			
			# Load lookup-table
			interpolator = self.correlationFunctionInterpolator(Pk_source)
				
			# If the requested radius is outside the range, give a detailed error message.
			R_req = numpy.min(R)
			if R_req < self.R_xi[0]:
				msg = 'R = %.2e is too small (min. R = %.2e)' % (R_req, self.R_xi[0])
				raise Exception(msg)
		
			R_req = numpy.max(R)
			if R_req > self.R_xi[-1]:
				msg = 'R = %.2e is too large (max. R = %.2e)' % (R_req, self.R_xi[-1])
				raise Exception(msg)
	
			# Interpolate to get xi(R). Note that the interpolation is performed in linear 
			# space, since xi can be negative.
			if derivative:
				ret = interpolator(R, nu = 1)
			else:
				ret = interpolator(R)
			
		else:

			if derivative:
				raise Exception('Derivative of xi cannot be evaluated if interpolation == False.')

			if Utilities.isArray(R):
				ret = R * 0.0
				for i in range(len(R)):
					ret[i] = self.correlationFunctionExact(R[i], Pk_source = Pk_source)
			else:
				ret = self.correlationFunctionExact(R, Pk_source = Pk_source)

		if not derivative and z > 1E-5:
			ret *= self.growthFactor(z)**2
		
		return	ret

	###############################################################################################
	# Peak curvature routines
	###############################################################################################
	
	# Get the mean peak curvature, <x>, at fixed nu from the integral of Bardeen et al. 1986 
	# (BBKS). Note that this function is approximated very well by the peakCurvatureApprox() 
	# function below.
	
	def peakCurvatureExact(self, nu, gamma):
	
		# Equation A15 in BBKS. 
		
		def curvature_fx(x):
	
			f1 = math.sqrt(5.0 / 2.0) * x
			t1 = scipy.special.erf(f1) + scipy.special.erf(f1 / 2.0)
	
			b0 = math.sqrt(2.0 / 5.0 / math.pi)
			b1 = 31.0 * x ** 2 / 4.0 + 8.0 / 5.0
			b2 = x ** 2 / 2.0 - 8.0 / 5.0
			t2 = b0 * (b1 * math.exp(-5.0 * x ** 2 / 8.0) + b2 * math.exp(-5.0 * x ** 2 / 2.0))
	
			res = (x ** 3 - 3.0 * x) * t1 / 2.0 + t2
	
			return res
	
		# Equation A14 in BBKS, minus the normalization which is irrelevant here. If we need the 
		# normalization, the Rstar parameter also needs to be passed.
		
		def curvature_Npk(x, nu, gamma):
	
			#norm = math.exp(-nu**2 / 2.0) / (2 * math.pi)**2 / Rstar**3
			norm = 1.0
			fx = curvature_fx(x)
			xstar = gamma * nu
			g2 = 1.0 - gamma ** 2
			exponent = -(x - xstar) ** 2 / (2.0 * g2)
			res = norm * fx * math.exp(exponent) / math.sqrt(2.0 * math.pi * g2)
	
			return res
	
		# Average over Npk
		
		def curvature_Npk_x(x, nu, gamma):
			return curvature_Npk(x, nu, gamma) * x
	
		args = nu, gamma
		norm, _ = scipy.integrate.quad(curvature_Npk, 0.0, numpy.infty, args, epsrel = 1E-10)
		integ, _ = scipy.integrate.quad(curvature_Npk_x, 0.0, numpy.infty, args, epsrel = 1E-10)
		xav = integ / norm
	
		return xav
	
	###############################################################################################
	
	# Wrapper for the function above which takes tables of sigmas. This form can be more convenient 
	# when computing many different nu's. 
	
	def peakCurvatureExactFromSigma(self, sigma0, sigma1, sigma2, deltac_const = True):
	
		nu = self.collapseOverdensity(deltac_const, sigma0) / sigma0
		gamma = sigma1 ** 2 / sigma0 / sigma2
	
		x = nu * 0.0
		for i in range(len(nu)):
			x[i] = self.peakCurvatureExact(nu[i], gamma[i])
	
		return nu, gamma, x
	
	###############################################################################################
	
	# Get peak curvature from the approximate formula in BBKS. This approx. is excellent over the 
	# relevant range of nu.
	
	def peakCurvatureApprox(self, nu, gamma):
	
		# Compute theta according to Equation 6.14 in BBKS
		g = gamma
		gn = g * nu
		theta1 = 3.0 * (1.0 - g ** 2) + (1.216 - 0.9 * g ** 4) * numpy.exp(-g * gn * gn / 8.0)
		theta2 = numpy.sqrt(3.0 * (1.0 - g ** 2) + 0.45 + (gn / 2.0) ** 2) + gn / 2.0
		theta = theta1 / theta2
	
		# Equation 6.13 in BBKS
		x = gn + theta
		
		# Equation 6.15 in BBKS
		nu_tilde = nu - theta * g / (1.0 - g ** 2)
	
		return theta, x, nu_tilde
	
	###############################################################################################
	
	# Wrapper for the function above which takes tables of sigmas. This form can be more convenient 
	# when computing many different nu's. For convenience, various intermediate numbers are 
	# returned as well.
	
	def peakCurvatureApproxFromSigma(self, sigma0, sigma1, sigma2, deltac_const = True):
	
		nu = self.collapseOverdensity(deltac_const, sigma0) / sigma0
		gamma = sigma1**2 / sigma0 / sigma2
		
		theta, x, nu_tilde = self.peakCurvatureApprox(nu, gamma)
		
		return nu, gamma, x, theta, nu_tilde
	
	###############################################################################################
	
	# Get the average peak curvature, <x>, from a mass (in Msun / h) and redshift. The returned 
	# values depend on whether curvature is evaluated exactly or approximately:
	#
	# exact: 	nu, gamma, x
	# approx.:	nu, gamma, x, theta, nu_tilde
	#
	# See BBKS for the meaning of these parameters.
	
	def peakCurvature(self, M, z, filt = 'gaussian', Pk_source = 'eh98', \
					deltac_const = True, exact = False):
		
		R = self.lagrangianR(M)
		sigma0 = self.sigma(R, j = 0, z = z, filt = filt, Pk_source = Pk_source)
		sigma1 = self.sigma(R, j = 1, z = z, filt = filt, Pk_source = Pk_source)
		sigma2 = self.sigma(R, j = 2, z = z, filt = filt, Pk_source = Pk_source)
	
		if exact:
			return self.peakCurvatureExactFromSigma(sigma0, sigma1, sigma2, deltac_const = deltac_const)
		else:
			return self.peakCurvatureApproxFromSigma(sigma0, sigma1, sigma2, deltac_const = deltac_const)

###################################################################################################
# Setter / getter functions for cosmologies
###################################################################################################

# cosmo is set as the global cosmology, and can be obtained by calling getCurrent().

def setCurrent(cosmo):
	
	global current_cosmo
	current_cosmo = cosmo
	
	return cosmo

###################################################################################################

# Return the current global cosmology. If no cosmology has been set yet, raise an exception.

def getCurrent():
	
	if current_cosmo == None:
		raise Exception('Cosmology is not set.')

	return current_cosmo

###################################################################################################

# Set a named cosmology. The name has to be...
#
# 1) In the cosmologies dictionary defined at the top of this file. Parameters from this cosmology
#    can be overwritten with the params dictionary (for example, settings such as interpolation
#    and storage).
# 2) powerlaw_**** with a slope as the final digits, e.g. powerlaw_-2.61
# 3) Some other name, and the user passes a parameter dictionary like the ones listed at the top
#    of this file.
#
# The chosen cosmology is returned, and also set as global.

def setCosmology(cosmo_name, params = None):
	
	if 'powerlaw_' in cosmo_name:
		n = float(cosmo_name.split('_')[1])
		param_dict = cosmologies['powerlaw']
		param_dict['power_law'] = True
		param_dict['power_law_n'] = n
	elif cosmo_name in cosmologies:		
		param_dict = cosmologies[cosmo_name]
		if params != None:
			param_dict = dict(param_dict.items() + params.items())
	else:
		if params != None:
			param_dict = params.copy()
		else:
			msg = "Invalid cosmology (%s)." % (cosmo_name)
			raise Exception(msg)
		
	param_dict['name'] = cosmo_name
	cosmo = Cosmology(**(param_dict))
	setCurrent(cosmo)
	
	return cosmo

###################################################################################################

# Add a cosmology definition to the cosmologies dictionary. This cosmology can then be selected
# using the setCosmology() function above.

def addCosmology(cosmo_name, params):
	
	cosmologies[cosmo_name] = params
	
	return 

###################################################################################################
