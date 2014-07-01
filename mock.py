

import read_nbody
import halo_occupation as ho
import numpy as np

class HOD_mock(object):
	'''Base class for any HOD-based mock object'''

	def __init__(self,hod_dict=None):
		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		temp_halos = read_nbody.load_bolshoi_host_halos_fits()
		# create a dictonary of numpy arrays containing relevant halo information
		self.halos = {}
		self.halos['logM'] = np.array(np.log10(temp_halos.MVIR))
		self.halos['conc'] = np.array(temp_halos.RVIR/temp_halos.RS)

		# create a dictionary containing the HOD parameters
		# does not seem to behave as I want:
		# when I instantiate a new mock with a passed hod_dict, 
		# this seems to over-write the hod_dict values of the previously instantiation
		self.hod_dict = hod_dict
		if hod_dict is None:
			self.hod_dict = {}
			self.hod_dict['logMmin_cen'] = 11.68
			self.hod_dict['sigma_logM'] = 0.15
			self.hod_dict['logMmin_sat'] = 11.88
			self.hod_dict['Msat_ratio'] = 21.0
			self.hod_dict['alpha_sat'] = 1.02

		self.halos['ncen'] = ho.num_ncen(self.halos['logM'],self.hod_dict)
		self.halos['nsat'] = ho.num_nsat(self.halos['logM'],self.hod_dict)

				# create a dictionary of numpy arrays containing mock galaxies
		self.galaxies = {}
		self.galaxies['num_gals'] = np.sum(self.halos['ncen']) + np.sum(self.halos['nsat'])
		self.galaxies['satellite_fraction'] = (1.0*np.sum(self.halos['nsat']))/self.galaxies['num_gals']






