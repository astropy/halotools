###################################################################################################
#
# CosmologyDemo.py	 		(c) Benedikt Diemer
#							University of Chicago
#     				    	bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the Cosmology.py module. Uncomment the functions listed
# in main() to explore various aspects of the Cosmology module.
#
###################################################################################################

import numpy
import Utilities
import Cosmology

###################################################################################################

def main():
	
	# ---------------------------------------------------------------------------------------------
	# Basic operations: setting, getting and changing the cosmology
	# ---------------------------------------------------------------------------------------------

	#demonstrateSettingAndGetting()
	#demonstrateAdding()
	#demonstrateChanging()
	
	# ---------------------------------------------------------------------------------------------
	# Basic computations; nothing will be persistently stored between runs.
	# ---------------------------------------------------------------------------------------------

	#compute()

	# ---------------------------------------------------------------------------------------------
	# Advanced computations; these can take a while the first time they are executed, but the 
	# results are stored in lookup tables. The second execution should be lightning fast.
	# ---------------------------------------------------------------------------------------------

	#computeAdvanced()

	return

###################################################################################################

def printCosmologyName():
	
	cosmo = Cosmology.getCurrent()
	print((cosmo.name))
	
	return

###################################################################################################

def demonstrateSettingAndGetting():
	
	print("Let's set a cosmology and print it's name.")
	cosmo = Cosmology.setCosmology('WMAP9')
	print((cosmo.name))
	Utilities.printLine()
	print("Now we do the same but in a function, using the global cosmology variable.")
	printCosmologyName()
	Utilities.printLine()
	print("Now let's temporarily switch cosmology without destroying the cosmology objects.")
	old_cosmo = cosmo
	cosmo = Cosmology.setCosmology('planck1')
	printCosmologyName()
	Cosmology.setCurrent(old_cosmo)
	printCosmologyName()

	return

###################################################################################################

def setMyCosmo():
	
	cosmo = Cosmology.setCosmology('my_cosmo')
	
	return cosmo

###################################################################################################

def demonstrateAdding():

	my_cosmo = {'flat': True, 'H0': 72.0, 'Om0': 0.25, 'Ob0': 0.043, 'sigma8': 0.8, 'ns': 0.97}
	
	print("Let's set a non-standard cosmology")
	cosmo = Cosmology.setCosmology('my_cosmo', my_cosmo)
	print(("We are now in " + cosmo.name + ", H0 = %.1f" % (cosmo.H0)))
	Utilities.printLine()
	print("We can also add this cosmology to the library of cosmologies, which allos us to set it from any function.")
	Cosmology.addCosmology('my_cosmo', my_cosmo)
	cosmo = setMyCosmo()
	print(("We are once again in " + cosmo.name + ", H0 = %.1f" % (cosmo.H0)))
	
	return

###################################################################################################

def demonstrateChanging():
	
	cosmo = Cosmology.setCosmology('planck1')
	print(("We are in the " + cosmo.name + " cosmology"))
	print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))
	Utilities.printLine()
	print("Let's do something bad and change a parameter without telling the Cosmology class...")
	cosmo.Om0 = 0.27
	print("Now the universe is not flat any more:")
	print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))
	Utilities.printLine()
	print("Now let's do it correctly and call checkForChangedCosmology():")
	cosmo.checkForChangedCosmology()
	print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))
	print("It is OK to change parameters at any time, but you MUST call checkForChangedCosmology() immediately afterwards!")
	
	return

###################################################################################################

def compute():
	
	cosmo = Cosmology.setCosmology('WMAP9')
	z = numpy.array([0.0, 1.0, 10.0])
	
	print("All cosmology functions can be called with numbers or numpy arrays:")
	print(("z                 = " + str(z)))
	Utilities.printLine()
	print("Times are output in Gyr, for example:")
	print(("Age               = " + str(cosmo.age(z))))
	Utilities.printLine()
	print("Distances are output in Mpc/h, for example:")
	print(("Comoving distance = " + str(cosmo.comovingDistance(z_max = z))))
	Utilities.printLine()
	print("Densities are output in astronomical units, Msun h^2 / kpc^3, for example:")
	print(("Critical density  = " + str(cosmo.criticalDensity(z))))

	return

###################################################################################################

def computeAdvanced():

	cosmo = Cosmology.setCosmology('WMAP9')
	z = 0.0
	M = numpy.array([1E9, 1E12, 1E15])
	
	print("We are now executing a function that needs sigma(R).")
	Utilities.printLine()
	nu = cosmo.M_to_nu(M, z)
	print(("Peak height = " + str(nu)))
	Utilities.printLine()
	print("Now, sigma(R) should be stored in a binary file in the Data/ directory.")
	print("If you call this function again, it should execute much, much faster!")
	
	return

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
