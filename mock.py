

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
		1) a numpy record array of dark matter host halos, 
		2) a dictionary of HOD model parameters,
		3) a numpy record array of galaxies populating those halos according to the model.


	'''

	def __init__(self,hod_dict=None,color_dict=None):

		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		simulation_data = read_nbody.load_bolshoi_host_halos_fits()
		temp_halos = simulation_data['halos']

		# create a dictonary of numpy arrays containing relevant halo information		
		halo_data_structure=[('logM','f4'),('conc','f4'),('ID','i8'),('pos','3float32'),('vel','3float32'),('rvir','f4'),('ncen','i4'),('nsat','i4')]
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

		self.halos['ncen']=np.array(ho.num_ncen(self.halos['logM'],self.hod_dict))
		self.halos['nsat']=np.array(ho.num_nsat(self.halos['logM'],self.hod_dict))
		self.ngals = np.sum(self.halos['ncen']) + np.sum(self.halos['nsat'])
		self.satellite_fraction = 1.0*np.sum(self.halos['nsat'])/(1.0*self.ngals)

		galaxy_data_structure=[('logM','f4'),('conc','f4'),('haloID','i8'),('pos','3float32'),('vel','3float32'),('hostpos','3float32'),('hostvel','3float32'),('rvir','f4'),('icen','i2'),('ired','i2')]
		self.galaxies = np.zeros(self.ngals,dtype=galaxy_data_structure)
		
		# Assign properties to the centrals
		self.galaxies['logM'][:np.sum(self.halos['ncen'])] = self.halos['logM'][(self.halos['ncen']>0)]
		self.galaxies['conc'][:np.sum(self.halos['ncen'])] = self.halos['conc'][(self.halos['ncen']>0)]
		self.galaxies['haloID'][:np.sum(self.halos['ncen'])] = self.halos['ID'][(self.halos['ncen']>0)]
		self.galaxies['pos'][:np.sum(self.halos['ncen'])] = self.halos['pos'][(self.halos['ncen']>0)]
		self.galaxies['hostpos'][:np.sum(self.halos['ncen'])] = self.halos['pos'][(self.halos['ncen']>0)]
		self.galaxies['vel'][:np.sum(self.halos['ncen'])] = self.halos['vel'][(self.halos['ncen']>0)]
		self.galaxies['hostvel'][:np.sum(self.halos['ncen'])] = self.halos['vel'][(self.halos['ncen']>0)]
		self.galaxies['rvir'][:np.sum(self.halos['ncen'])] = self.halos['rvir'][(self.halos['ncen']>0)]
		self.galaxies['icen'][:np.sum(self.halos['ncen'])] = np.zeros(np.sum(self.halos['ncen']))+1
		









