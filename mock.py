

import read_nbody
import halo_occupation as ho
import numpy as np

class HOD_mock(object):
	'''Base class for any HOD-based mock object'''

	def __init__(self):
		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		temp_halos = read_nbody.load_bolshoi_host_halos_fits()
		# create a dictonary of numpy arrays containing relevant halo information
		self.halos = {}
		self.halos['logM'] = np.array(np.log10(temp_halos.MVIR))
		self.halos['conc'] = np.array(temp_halos.RVIR/temp_halos.RS)

		# create a dictionary of numpy arrays containing mock galaxies
		self.galaxies = {}

		# create a dictionary containing the HOD parameters
		self.hod_dict = {}
		self.hod_dict['logMmin_cen'] = 12.0
		self.hod_dict['sigma_logM'] = 0.2
		self.hod_dict['logMmin_sat'] = 12.25
		self.hod_dict['Msat_ratio'] = 20.0
		self.hod_dict['alpha_sat'] = 1.0









