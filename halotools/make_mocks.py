
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

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
import astropy


def apply_periodicity_of_box(coords, box_length):
    """

    Parameters
    ----------
    coords: 1d list of floats
    box_length : scalar giving size of simulation box (currently hard-coded to be Mpc/h units)

	Returns
    -------
    coords : 1d list of floats corrected for box periodicity

    """
    test = coords > box_length
    coords[test] = coords[test] - box_length
    test = coords < 0
    coords[test] = box_length + coords[test]
    return coords

def num_cen_monte_carlo(logM,hod_dict):
    """ Returns Monte Carlo-generated array of 0 or 1 specifying whether there is a central in the halo.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary

    Returns
    -------
    num_ncen_array : int or array

    
    """

    num_ncen_array = np.array(ho.mean_ncen(logM,hod_dict) > np.random.random(len(logM)),dtype=int)
    return num_ncen_array

def num_nsat_monte_carlo(logM,hod_dict):
    '''  Returns Monte Carlo-generated array of integers specifying the number of satellites in the halo.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary

    Returns
    -------
    num_nsat_array : int or array
        Values of array specify the number of satellites hosted by each halo.


    '''
    Prob_sat = ho.mean_nsat(logM,hod_dict)
	# NOTE: need to cut at zero, otherwise poisson bails
    # BUG IN SCIPY: poisson.rvs bails if there are zeroes in a numpy array
    test = Prob_sat <= 0
    Prob_sat[test] = defaults.default_tiny_poisson_fluctuation

    num_nsat_array = poisson.rvs(Prob_sat)

    return num_nsat_array

def _generate_random_points_on_unit_sphere(Num_points):
	"""
	
	Parameters
    ----------
    Num_points : int
    	Specifies number of random points required

    Returns
    -------
    coords : 3 x Num_points numpy array of coordinate points on the unit sphere.

	"""
	
	phi = np.random.uniform(0,2*np.pi,Num_points)
	cos_t = np.random.uniform(-1.,1.,Num_points)
	sin_t = np.sqrt((1.-cos_t**2))
	coords = np.zeros(Num_points*3).reshape([Num_points,3])
	coords[:,0] = sin_t * np.cos(phi)
	coords[:,1] = sin_t * np.sin(phi)
	coords[:,2] = cos_t
	return coords	
	

class HOD_mock(object):
	'''		

	Parameters
	----------
	hod_dict : optional, dictionary containing parameter values specifying how to populated dark matter halos with mock galaxies
	color_dict : optional, dictionary containing parameter values specifying how colors are assigned to mock galaxies

	Synopsis
	--------
	Base class for any HOD-based mock galaxy catalog object. 
		Instantiations of this class have bound to them: 
		* a numpy record array of dark matter host halos, 
		* a dictionary of HOD model parameters,
		* a numpy record array of galaxies populating those halos according to the model.
		* methods for computing mock observables, such as (marked) two-point clustering, gg-lensing, conformity, etc. (yet to be implemented)


	Warning
	-------
	Still buggy and poorly tested. Basic API still under rapid development.
	Not yet suitable even for use at your own risk.


	'''

	def __init__(self,simulation_data,hod_dict=None,color_dict=None,seed=None):

		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		# eventually this step will not require a "pre-processed" halo catalog, but this if fine for now.

#		if not isinstance(simulation_data['halos'],astropy.table.table.Table):
#			raise TypeError("HOD_mock object requires an astropy Table halo catalog as input")
		table_of_halos = simulation_data['halos']

		if not isinstance(simulation_data['simulation_dict'],dict):
			raise TypeError("HOD_mock object requires a dictionary of simulation metadata as input")
		self.simulation_dict = simulation_data['simulation_dict']

		self.logM = np.array(np.log10(table_of_halos['MVIR']))
		self.haloID = np.array(table_of_halos['ID'])
		self.concen = ho.anatoly_concentration(self.logM)
		self.Rvir = np.array(table_of_halos['RVIR'])/1000.

		self.halopos = np.empty((len(table_of_halos),3),'f8')
		self.halopos.T[0] = np.array(table_of_halos['POS'][:,0])
		self.halopos.T[1] = np.array(table_of_halos['POS'][:,1])
		self.halopos.T[2] = np.array(table_of_halos['POS'][:,2])

		if seed != None:
			np.random.seed(seed)

		self.idx_current_halo = 0 # index for current halo (bookkeeping device to speed up array access)


		#Set up the grid used to tabulate NFW profiles
		#This will be used to assign halo-centric distances to the satellites
		Npts_concen = defaults.default_Npts_concentration_array
		concentration_array = np.linspace(self.concen.min(),self.concen.max(),Npts_concen)
		Npts_radius = defaults.default_Npts_radius_array		
		radius_array = np.linspace(0.,1.,Npts_radius)
		
		cumulative_nfw_PDF = []
		# After executing the following lines, 
		#cumulative_nfw_PDF will be a list of functions. 
		#The list elements correspond to functions governing 
		#radial profiles of halos with different NFW concentrations.
		#Each function takes a scalar y in [0,1] as input, 
		#and outputs the x = r/Rvir corresponding to Prob_NFW( x > r/Rvir ) = y. 
		#Thus each list element is the inverse of the NFW cumulative NFW PDF.
		for c in concentration_array:
			cumulative_nfw_PDF.append(interp1d(ho.cumulative_NFW_PDF(radius_array,c),radius_array))

		#interp_idx_concen is an integer array with one element per host halo
		#each element gives the index pointing to the bins defined by concentration_array
		self.interp_idx_concen = np.digitize(self.concen,concentration_array)


	def _setup(self,hod_dict=defaults.default_hod_dict):
		"""
		Compute NCen,Nsat and preallocate arrays for __call__().
		
		"""

		self.Ncen = num_cen_monte_carlo(self.logM,hod_dict)
		self.hasCentral = self.NCen > 0
		self.NSat = np.zeros(len(self.logM),dtype=int)
		self.NSat[self.hasCentral] = num_sat_monte_carlo(self.logM[self.hasCentral],hod_dict,output=self.Nsat[self.hasCentral])

		self.num_total_cens = self.NCen.sum()
		self.num_total_sats = self.NSat.sum()
		self.num_total_gals = self.num_total_cens + self.num_total_sats

		# preallocate output arrays
		self.coords = np.empty((self.num_total_gals,3),dtype='f8')
		self.logMhost = np.empty(self.num_total_gals,dtype='f8')
		self.isSat = np.zeros(self.num_total_gals,dtype='i4')
	#...

	def _random_angles(self,coords,start,end,N):
		"""
		Generate N random angles and assign them to coords[start:end].
		"""

		cos_t = np.random.uniform(-1.,1.,N)
		phi = np.random.uniform(0,2*np.pi,N)
		sin_t = np.sqrt((1.-cos_t**2))
		
		coords[start:end,0] = sin_t * np.cos(phi)
		coords[start:end,1] = sin_t * np.sin(phi)
		coords[start:end,2] = cos_t
	#...



	def __call__(self,hod_dict=defaults.default_hod_dict,isSetup=False):
		"""
		Return a list of galaxies placed randomly in the halos.
		Returns coordinates, halo mass, isSat (boolean array with True for satellites)
		If isSetup is True, don't call _setup first (useful for calling from a child class).

		"""

		# pregenerate the output arrays
		if not isSetup:
			self._setup(hod_dict)

		# Preallocate centrals so we don't have to touch them again.
		self.coords[:self.num_total_cens] = self.halopos[self.hasCentral]
		self.logMhost[:self.num_total_cens] = self.logM[self.hasCentral]

		counter = self.num_total_cens
		self.idx_current_halo = 0

		self.isSat[counter:] = 1 # everything else is a satellite.
		# Pregenerate satellite angles all in one fell swoop
		self._random_angles(self.coords,counter,self.coords.shape[0],self.num_total_sats)

		# all the satellites will now end up at the end of the array.
		satellite_index_array = np.nonzero(self.NSat > 0)[0]
		# these two save a bit of time by eliminating calls to records.__getattribute__
		logmasses = self.logM

		for self.ii in satellite_index_array:
			logM,center,Nsat,r_vir,cidx = logmasses[self.ii],self.halopos[self.ii],self.NSat[self.ii],self.Rvir[self.ii],self.interp_idx_concen[self.ii]
			self.logMhost[counter:counter+Nsat] = logM
			counter += Nsat


		x=4

		return x
	#...

















"""
		# create a numpy record array containing halo information relevant to this class of HODs	
		halo_data_structure=[
			('logM','f4'),('conc','f4'),('ID','i8'),
			('pos','3float32'),('vel','3float32'),('rvir','f4'),
			('ncen','i4'),('nsat','i4')
			]

		self.halos = (np.zeros(len(table_of_halos['MVIR']),dtype=halo_data_structure))
		self.halos['logM'] = np.log10(table_of_halos['MVIR'])
		self.halos['conc'] = table_of_halos['RVIR']/table_of_halos['RS']
		self.halos['ID'] = table_of_halos['ID']
		self.halos['pos'] = table_of_halos['POS']
		self.halos['vel'] = table_of_halos['VEL']
		self.halos['rvir'] = np.array(table_of_halos['RVIR'])/1000.

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

		# add some convenience tags to the halos array as well as the mock object
		self.halos['ncen']=np.array(ho.num_ncen(self.halos['logM'],self.hod_dict))
		self.halos['nsat']=np.array(ho.num_nsat(self.halos['logM'],self.hod_dict))
		self.ngals = np.sum(self.halos['ncen']) + np.sum(self.halos['nsat'])
		self.nsats = np.sum(self.halos['nsat'])
		self.ncens = np.sum(self.halos['ncen'])
		self.satellite_fraction = 1.0*np.sum(self.halos['nsat'])/(1.0*self.ngals)

		# create a numpy record array containing galaxy catalog from which 
		# properties of the mock object derive 	
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
		
		# Assign host halo properties to the satellites
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

			


		satellite_indices = np.nonzero(self.galaxies['icen']==0)[0]
		random_numbers_for_satellite_positions = np.random.random(len(satellite_indices))
		#Looping over each individual satellite adds an entire second to the runtime.
		#This is unacceptable. Make it faster! But the bookkeeping is correct, so this is fine for now.
		#Since the total runtime for mock creation is just 2 seconds, 
		#this will suffice until it's time to implement the likelihood engine.
		for ii in np.arange(len(satellite_indices)):
			self.galaxies['rhalo'][satellite_indices[ii]] = cumulative_nfw_PDF[idx_conc[ii]](random_numbers_for_satellite_positions[ii])

		#Assign positions to satellites by randomly drawing host-centric positions from an NFW profile
		self.galaxies['pos'][:,0][self.galaxies['icen']==0] = self.galaxies['hostpos'][:,0][self.galaxies['icen']==0] + (_generate_random_points_on_unit_sphere(self.nsats)[:,0]*self.galaxies['rhalo'][self.galaxies['icen']==0]*self.galaxies['rvir'][self.galaxies['icen']==0])
		self.galaxies['pos'][:,1][self.galaxies['icen']==0] = self.galaxies['hostpos'][:,1][self.galaxies['icen']==0] + (_generate_random_points_on_unit_sphere(self.nsats)[:,1]*self.galaxies['rhalo'][self.galaxies['icen']==0]*self.galaxies['rvir'][self.galaxies['icen']==0])
		self.galaxies['pos'][:,2][self.galaxies['icen']==0] = self.galaxies['hostpos'][:,2][self.galaxies['icen']==0] + (_generate_random_points_on_unit_sphere(self.nsats)[:,2]*self.galaxies['rhalo'][self.galaxies['icen']==0]*self.galaxies['rvir'][self.galaxies['icen']==0])
#		self.galaxies[self.galaxies['icen']==0]['pos'][:,0] = self.galaxies[self.galaxies['icen']==0]['hostpos'][:,0] + (_generate_random_points_on_unit_sphere(self.nsats)[:,0]*self.galaxies[self.galaxies['icen']==0]['rhalo']*self.galaxies[self.galaxies['icen']==0]['rvir'])
#		self.galaxies[self.galaxies['icen']==0]['pos'][:,1] = self.galaxies[self.galaxies['icen']==0]['hostpos'][:,1] + (_generate_random_points_on_unit_sphere(self.nsats)[:,1]*self.galaxies[self.galaxies['icen']==0]['rhalo']*self.galaxies[self.galaxies['icen']==0]['rvir'])
#		self.galaxies[self.galaxies['icen']==0]['pos'][:,2] = self.galaxies[self.galaxies['icen']==0]['hostpos'][:,2] + (_generate_random_points_on_unit_sphere(self.nsats)[:,2]*self.galaxies[self.galaxies['icen']==0]['rhalo']*self.galaxies[self.galaxies['icen']==0]['rvir'])

		
		#Apply correction to account for the periodic box
		self.galaxies['pos'][:,0] = apply_periodicity_of_box(self.galaxies['pos'][:,0],self.simulation_dict['Lbox'])
		self.galaxies['pos'][:,1] = apply_periodicity_of_box(self.galaxies['pos'][:,1],self.simulation_dict['Lbox'])
		self.galaxies['pos'][:,2] = apply_periodicity_of_box(self.galaxies['pos'][:,2],self.simulation_dict['Lbox'])


"""


#def _generate_random_points_on_unit_sphere(self,Num_points):


		
#	def _assign_satellite_coords_on_virial_sphere(self):
#		satellite_coords_on_unit_sphere = self._generate_random_points_on_unit_sphere(self.galaxies.nsats)
#		for idim in np.arange(3):
#			self.galaxies['pos'][self.galaxies['icen']==0,idim]=satellite_coords_on_unit_sphere[:,idim]
#		self.galaxies['pos'][self.galaxies['icen']==0,:] *= self.galaxies['rvir']/1000.0
#		self.galaxies['pos'][self.galaxies['icen']==0,:] += self.galaxies['hostpos'][self.galaxies['icen']==0,:]
		












