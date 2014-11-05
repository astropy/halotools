###################################################################################################
#
# ConcentrationDemo.py 		(c) Benedikt Diemer
#							University of Chicago
#     				    	bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the HaloDensityProfile.py module. 
#
###################################################################################################

import Cosmology
import HaloDensityProfile

###################################################################################################

def main():

	demonstrateMassDefinitions()

	return

###################################################################################################

# Convert one mass definition to another, assuming an NFW profile

def demonstrateMassDefinitions():
	
	Mvir = 1E12
	cvir = 10.0
	z = 0.0
	Cosmology.setCosmology('WMAP9')

	Rvir = HaloDensityProfile.R_Delta(Mvir, z, 'vir')
	
	print(("Mvir:   %.2e Msun / h" % Mvir))
	print(("Rvir:   %.2e kpc / h" % Rvir))
	print(("cvir:   %.2f" % cvir))
	
	M200c, R200c, c200c = HaloDensityProfile.convertMassDefinition(Mvir, cvir, z, 'vir', '200c')
	
	print(("M200c:  %.2e Msun / h" % M200c))
	print(("R200c:  %.2e kpc / h" % R200c))
	print(("c200c:  %.2f" % c200c))
	
	return

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
