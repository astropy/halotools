

import read_nbody
import halo_occupation as ho
import numpy as np
from scipy.integrate import quad as quad
from scipy.interpolate import interp1d as interp1d
import defaults
import cPickle
import os
from copy import copy
from collections import Counter


def rewrap(coords, box_length):
    """Rewrap coords to all be within the box. Returns the rewrapped result."""
    test = coords > box_length
    coords[test] = coords[test] - box_length
    test = coords < 0
    coords[test] = box_length + coords[test]
    return coords

	
def _generate_random_points_on_unit_sphere(self,Num_points):
	"""Generate N random angles and assign them to coords[start:end]."""
	phi = np.random.uniform(0,2*np.pi,Num_points)
	cos_t = np.random.uniform(-1.,1.,Num_points)
	sin_t = np.sqrt((1.-cos_t**2))
	coords = np.zeros(Num_points*3).reshape([Num_points,3])
	coords[:,0] = sin_t * np.cos(phi)
	coords[:,1] = sin_t * np.sin(phi)
	coords[:,2] = cos_t
	return coords	
	

class HOD_mock(object):
	'''Base class for any HOD-based mock galaxy catalog object.

	.. warning::
		Still buggy and poorly tested.

	Args:
		hod_dict : dictionary containing parameter values specifying how to populated dark matter halos with mock galaxies

	Synopsis:
		Instantiations of this class have bound to them: 
		* a numpy record array of dark matter host halos, 
		* a dictionary of HOD model parameters,
		* a numpy record array of galaxies populating those halos according to the model.


	'''

	def __init__(self,hod_dict=None,color_dict=None):

		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		simulation_data = read_nbody.load_bolshoi_host_halos_fits()
		temp_halos = simulation_data['halos']

		# create a dictonary of numpy arrays containing relevant halo information		
		halo_data_structure=[
			('logM','f4'),('conc','f4'),('ID','i8'),
			('pos','3float32'),('vel','3float32'),('rvir','f4'),
			('ncen','i4'),('nsat','i4')
			]
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
		self.nsats = np.sum(self.halos['nsat'])
		self.ncens = np.sum(self.halos['ncen'])
		self.satellite_fraction = 1.0*np.sum(self.halos['nsat'])/(1.0*self.ngals)

		galaxy_data_structure=[
			('logM','f4'),('conc','f4'),('haloID','i8'),
			('pos','3float32'),('vel','3float32'),('hostpos','3float32'),
			('hostvel','3float32'),('rvir','f4'),('icen','i2'),
			('ired','i2'),('rhalo','f4')
			]
		self.galaxies = np.zeros(self.ngals,dtype=galaxy_data_structure)
		
		# Assign properties to the centrals
		self.galaxies['logM'][0:self.ncens] = self.halos['logM'][(self.halos['ncen']>0)]
		self.galaxies['haloID'][0:self.ncens] = self.halos['ID'][(self.halos['ncen']>0)]
		self.galaxies['pos'][0:self.ncens] = self.halos['pos'][(self.halos['ncen']>0)]
		self.galaxies['hostpos'][0:self.ncens] = self.halos['pos'][(self.halos['ncen']>0)]
		self.galaxies['vel'][0:self.ncens] = self.halos['vel'][(self.halos['ncen']>0)]
		self.galaxies['hostvel'][0:self.ncens] = self.halos['vel'][(self.halos['ncen']>0)]
		self.galaxies['rvir'][0:self.ncens] = self.halos['rvir'][(self.halos['ncen']>0)]
		self.galaxies['icen'][0:self.ncens] = np.zeros(np.sum(self.halos['ncen']))+1
		self.galaxies['rhalo'][0:self.ncens] = np.zeros(np.sum(self.halos['ncen']))
		
		# Assign host properties to the satellites
		# Currently involves looping over every host halo with Nsat > 0
		# This is one of the potential speed bottlenecks
		counter=np.sum(self.halos['ncen'])
		halos_with_satellites = self.halos[self.halos['nsat']>0]
		for halo in halos_with_satellites:
			self.galaxies['logM'][counter:counter+halo['nsat']] = halo['logM']
			self.galaxies['haloID'][counter:counter+halo['nsat']] = halo['ID']
			self.galaxies['hostpos'][counter:counter+halo['nsat']] = halo['pos']
			self.galaxies['hostvel'][counter:counter+halo['nsat']] = halo['vel']
			self.galaxies['rvir'][counter:counter+halo['nsat']] = halo['rvir']
			counter += halo['nsat']

		#over-write the true halo concentrations with Anatoly's best-fit relation
		#this erases the relationship between halo assembly and internal structure, 
		#in accord with the conventions of "mass-only" HODs
		self.galaxies['conc'] = ho.anatoly_concentration(self.galaxies['logM'])*self.hod_dict['fconc']
		concentration_array = np.linspace(np.min(self.galaxies['conc']),np.max(self.galaxies['conc']),1000)
		radius_array = np.linspace(0.,1.,101)
		#cumulative_nfw_PDF is a list of functions. The list elements correspond to different NFW concentrations.
		#Each function takes a scalar y in [0,1] as input, 
		#and outputs the x = r/Rvir corresponding to P_NFW( x > r/Rvir ) = y. 
		#Thus each list element is the inverse of the NFW cumulative mass PDF.
		cumulative_nfw_PDF = []
		for c in concentration_array:
			cumulative_nfw_PDF.append(interp1d(ho.cumulative_NFW_PDF(radius_array,c),radius_array))
		#idx_conc is an integer array, with one element for each satellite in self.galaxies
		#the elements of idx_conc are the indices pointing to the bins defined by concentration_array
		idx_conc = np.digitize(self.galaxies['conc'][self.ncens:],concentration_array)
		satellite_rhalo_array = np.random.random(self.nsats)

		satellite_indices = np.nonzero(self.galaxies['icen']==0)[0]
		random_numbers_for_satellite_positions = np.random.random(len(satellite_indices))
		#Looping over each individual satellite adds an entire second to the runtime.
		#This is unacceptable. Make it faster!
		for ii in np.arange(len(satellite_indices)):
			self.galaxies['rhalo'][satellite_indices[ii]] = cumulative_nfw_PDF[idx_conc[ii]](random_numbers_for_satellite_positions[ii])



		
	def _assign_satellite_coords_on_virial_sphere(self):
		satellite_coords_on_unit_sphere = self._generate_random_points_on_unit_sphere(self.galaxies.nsats)
		for idim in np.arange(3):
			self.galaxies['pos'][self.galaxies['icen']==0,idim]=satellite_coords_on_unit_sphere[:,idim]
#		self.galaxies['pos'][self.galaxies['icen']==0,:] *= self.galaxies['rvir']/1000.0
#		self.galaxies['pos'][self.galaxies['icen']==0,:] += self.galaxies['hostpos'][self.galaxies['icen']==0,:]
		












