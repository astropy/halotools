

import read_nbody
import halo_occupation as ho
import numpy as np
import defaults

class HOD_mock(object):
	'''Base class for any HOD-based mock galaxy catalog object.

	.. warning::
		Still buggy and poorly tested.

	Args:
		hod_dict : dictionary containing parameter values specifying how to populated dark matter halos with mock galaxies

	Synopsis:
		Instantiations of this class have bound to them: 
		1) a dictionary of dark matter host halos, 
		2) a dictionary of HOD model parameters,
		3) a dictionary of galaxies populating those halos according to the model.


	'''

	def __init__(self,hod_dict=None,color_dict=None):

		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		simulation_data = read_nbody.load_bolshoi_host_halos_fits()
		temp_halos = simulation_data['halos']

		# create a dictonary of numpy arrays containing relevant halo information		
		halo_data_structure=[('logM','f4'),('conc','f4'),('ID','i8'),('pos','3float32'),('vel','3float32'),('rvir','f4')]
		self.halos = np.zeros(len(temp_halos.MVIR),dtype=halo_data_structure)				
		self.halos['logM'] = np.log10(temp_halos.MVIR)
		self.halos['conc'] = temp_halos.RVIR/temp_halos.RS
		self.halos['ID'] = temp_halos.ID
		self.halos['pos'] = temp_halos.POS
		self.halos['vel'] = temp_halos.VEL
		self.halos['rvir'] = np.array(temp_halos.RVIR)

		# mock object should know the basic attributs of its simulation
		self.simulation_dict = simulation_data['simulation_dict']
		

		# create a dictionary containing the HOD parameters
		if hod_dict is None:
			self.hod_dict = defaults.default_hod_dict
		else:
			self.hod_dict = hod_dict

		if color_dict is None:
			self.color_dict = defaults.default_color_dict
		else:
			self.color_dict = color_dict

		ncen_array = ho.num_ncen(self.halos['logM'],self.hod_dict)
		nsat_array = ho.num_nsat(self.halos['logM'],self.hod_dict)
		total_number_centrals = np.sum(ncen_array)
		total_number_satellites = np.sum(nsat_array)
		total_number_gals = total_number_centrals + total_number_satellites

				# create a dictionary of numpy arrays containing mock galaxies
		galaxy_data_structure=[('logM','f4'),('conc','f4'),('haloID','i8'),('pos','3float32'),('vel','3float32'),('rvir','f4'),('icen','i2'),('ired','i2')]
		self.galaxies = np.zeros(total_number_gals,dtype=galaxy_data_structure)


		self.satellite_fraction = 1.0*total_number_satellites/(1.0*total_number_gals)






