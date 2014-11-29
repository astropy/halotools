###################################################################################################
#
# HaloConcentrationDemo.py  (c) Benedikt Diemer
#							University of Chicago
#     				    	bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the HaloConcentration.py module. 
#
###################################################################################################

import numpy
import Utilities
import Cosmology
import HaloConcentration
import HaloDensityProfile

###################################################################################################

def main():
	
	computeConcentration()
	#computeConcentrationTable('WMAP9')
	#computeAllTables()

	return

###################################################################################################

# A small demonstration of the concentration model. This function outputs the concentrations in 
# the virial and 200c mass definitions, for three halo masses, in the WMAP9 cosmology.

def computeConcentration():
	
	M = numpy.array([1E9, 1E12, 1E15])

	print("First, set a cosmology")
	cosmo = Cosmology.setCosmology('WMAP9')
	print(("Cosmology is %s" % cosmo.name))
	
	Utilities.printLine()
	print("Now compute concentrations for M200c:")
	c = HaloConcentration.dk14_c200c_M(M, 0.0, statistic = 'median')
	for i in range(len(M)):
		print(("M200c = %.2e, c200c = %5.2f" % (M[i], c[i])))

	Utilities.printLine()
	print("Now compute concentrations for another mass definition, Mvir:")
	c = HaloConcentration.concentration(M, 'vir', 0.0, model = 'dk14', statistic = 'median')
	for i in range(len(M)):
		print(("Mvir = %.2e, cvir = %5.2f" % (M[i], c[i])))

	Utilities.printLine()
	print("We note that the prediction for mass definitions other than c200c is not as accurate")
	print("due to differences between the real density profiles and the NFW approximation that")
	print("is used for the conversion. See Appendix C of Diemer & Kravtsov 2014b for details.")
	
	return

###################################################################################################

# Create data tables for a range of cosmologies.

def computeAllTables():
	
	cosmos = ['bolshoi', 'millennium', 'planck1', 'planck1-only', 'WMAP9', 'WMAP9-ML', 'WMAP9-only', \
			'WMAP7', 'WMAP7-ML', 'WMAP7-only', 'WMAP5', 'WMAP5-ML', 'WMAP5-only']
	for c in cosmos:
		computeConcentrationTable(c)
		
	return

###################################################################################################

# Create a file with a table of concentrations for a range of redshifts, halo masses, and mass
# definitions.

def computeConcentrationTable(cosmo_name):
	
	cosmo = Cosmology.setCosmology(cosmo_name)
	mdefs = ['2500c', '500c', 'vir', '200m']
	nu_min = 0.3
	nu_max = 5.5
	n_M_bins = 100
	z = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]

	# Write file header
	f = open('cM_%s.txt' % (cosmo_name), 'w')
	line = '# This file contains concentrations according to the model of Diemer & Kravtsov 2014. The model natively predicts the mean and median c200c. For all other mass definitions,\n'
	f.write(line)
	line = '# the halo mass and concentrations are converted using either the mean or median c200c.\n'
	f.write(line)
	line = '#\n'
	f.write(line)
	line = '# Cosmology: ' + str(Cosmology.cosmologies[cosmo_name]) + '\n'
	f.write(line)
	line = '#\n'
	f.write(line)
	line = '#   z     nu     M200c  c200c  c200c'
	for k in range(len(mdefs)):
		for dummy in range(2):
			n_spaces = 9 - len(mdefs[k])
			for dummy in range(n_spaces):
				line += ' '
			line += 'M%s' % (mdefs[k])
			n_spaces = 6 - len(mdefs[k])
			for dummy in range(n_spaces):
				line += ' '
			line += 'c%s' % (mdefs[k])
	line += '\n'
	f.write(line)
	line = '#                      median   mean'
	for dummy in range(len(mdefs)):
		line += '    median median      mean   mean'
	line += '\n'
	f.write(line)
	
	# Write block for each redshift
	for i in range(len(z)):

		print z[i]

		if z[i] > 5.0:
			nu_min = 1.0
			
		log_M_min = numpy.log10(cosmo.nu_to_M(nu_min, z[i]))
		log_M_max = numpy.log10(cosmo.nu_to_M(nu_max, z[i]))
		bin_width_logM = (log_M_max - log_M_min) / float(n_M_bins - 1)
		
		M200c = 10**numpy.arange(log_M_min, log_M_max + bin_width_logM, bin_width_logM)
		M200c = M200c[:n_M_bins]	
		nu200c = cosmo.M_to_nu(M200c, z[i])
		c200c_median = HaloConcentration.dk14_c200c_nu(nu200c, z[i], statistic = 'median')
		c200c_mean = HaloConcentration.dk14_c200c_nu(nu200c, z[i], statistic = 'mean')
			
		for j in range(len(M200c)):
			line = '%5.2f  %5.3f  %8.2e  %5.2f  %5.2f' % (z[i], nu200c[j], M200c[j], c200c_median[j], c200c_mean[j])
			prof_median = HaloDensityProfile.NFWProfile(M = M200c[j], c = c200c_median[j], z = z[i], mdef = '200c')
			prof_mean = HaloDensityProfile.NFWProfile(M = M200c[j], c = c200c_mean[j], z = z[i], mdef = '200c')			
			for k in range(len(mdefs)):
				R_delta_median, M_delta_median = prof_median.RMDelta(z[i], mdefs[k])
				R_delta_mean, M_delta_mean = prof_mean.RMDelta(z[i], mdefs[k])
				c_delta_median = R_delta_median / prof_median.rs
				c_delta_mean = R_delta_mean / prof_mean.rs
				line += '  %8.2e  %5.2f  %8.2e  %5.2f' % (M_delta_median, c_delta_median, M_delta_mean, c_delta_mean)
			line += '\n'
			f.write(line)
			
	f.close()
	
	return

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
