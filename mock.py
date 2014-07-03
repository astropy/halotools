

import read_nbody
import halo_occupation as ho
import numpy as np
import defaults
import copy

class HOD_mock(object):
	'''Base class for any HOD-based mock galaxy catalog object.

	.. warning::
		Still buggy and poorly tested.

	Parameters
	----------
	hod_dict : dictionary
		Contains parameter values specifying how to populated dark matter halos
		with mock galaxies

	Notes
	----------
	Instantiations of this class have bound to them: 
	1) a dictionary of dark matter host halos, 
	2) a dictionary of HOD model parameters,
	3) a dictionary of galaxies populating those halos according to the model.


	'''

	def __init__(self,hod_dict=None):

		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		simulation = read_nbody.load_bolshoi_host_halos_fits()
		temp_halos = simulation['halos']

		# create a dictonary of numpy arrays containing relevant halo information
		self.halos = {}
		self.halos['logM'] = np.array(np.log10(temp_halos.MVIR))
		self.halos['conc'] = np.array(temp_halos.RVIR/temp_halos.RS)
		self.halos['ID'] = np.array(temp_halos.ID)
		self.halos['pos'] = np.array([temp_halos.POS[:,0],temp_halos.POS[:,1],temp_halos.POS[:,2]])
		self.halos['vel'] = np.array([temp_halos.VEL[:,0],temp_halos.VEL[:,1],temp_halos.VEL[:,2]])
		self.halos['rvir'] = np.array(temp_halos.RVIR)

		# mock object should know the basic attributs of its simulation
		self.simulation_dict = simulation['simulation_dict']

		# create a dictionary containing the HOD parameters
		if hod_dict is None:
			self.hod_dict = defaults.default_hod_dict
		else:
			self.hod_dict = hod_dict

		self.halos['ncen'] = ho.num_ncen(self.halos['logM'],self.hod_dict)
		self.halos['nsat'] = ho.num_nsat(self.halos['logM'],self.hod_dict)

				# create a dictionary of numpy arrays containing mock galaxies
		self.galaxies = {}
		self.galaxies['num_gals'] = np.sum(self.halos['ncen']) + np.sum(self.halos['nsat'])
		self.galaxies['satellite_fraction'] = (1.0*np.sum(self.halos['nsat']))/self.galaxies['num_gals']






