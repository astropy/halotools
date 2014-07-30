
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import read_nbody
import halo_occupation as ho
import numpy as np

from scipy.integrate import quad as quad
from scipy.interpolate import interp1d as interp1d
from scipy.stats import poisson

import defaults
import cPickle
import os
from copy import copy
from collections import Counter
import astropy


def enforce_periodicity_of_box(coords, box_length):
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

def num_cen_monte_carlo(logM,hod_model):
    """ Returns Monte Carlo-generated array of 0 or 1 specifying whether there is a central in the halo.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary

    Returns
    -------
    num_ncen_array : int or array

    
    """
    num_ncen_array = np.array(
    	hod_model.mean_ncen(logM) > np.random.random(len(logM)),dtype=int)

    return num_ncen_array

def num_sat_monte_carlo(logM,hod_model,output=None):
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
    Prob_sat = hod_model.mean_nsat(logM)
	# NOTE: need to cut at zero, otherwise poisson bails
    # BUG IN SCIPY: poisson.rvs bails if there are zeroes in a numpy array
    test = Prob_sat <= 0
    Prob_sat[test] = defaults.default_tiny_poisson_fluctuation

    num_nsat_array = poisson.rvs(Prob_sat)

    return num_nsat_array

def quenched_monte_carlo(logM,quenching_model,galaxy_type):
    """ Returns Monte Carlo-generated array of 0 or 1 specifying whether the galaxy is quenched.

    Parameters
    ----------
    quenched_fractions : array of expectation values for quenching

    Returns
    -------
    quenched_array : int or array

    
    """
    if galaxy_type == 'central':
    	quenching_function = quenching_model.mean_quenched_fraction_centrals
    elif galaxy_type == 'satellite':
    	quenching_function = quenching_model.mean_quenched_fraction_satellites
    else:
    	raise TypeError("input galaxy_type must be a string set either to 'central' or 'satellite'")


    quenched_array = np.array(
    	quenching_function(logM) > np.random.random(
    		len(logM)),dtype=int)

    return quenched_array


	

class HOD_mock(object):
	'''		

	Parameters
	----------
	hod_model : optional, dictionary containing parameter values specifying how to populated dark matter halos with mock galaxies
	quenching_model : optional, dictionary containing parameter values specifying how colors are assigned to mock galaxies

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

	def __init__(self,simulation_data,hod_model,quenching_model,seed=None):

		# read in .fits file containing pre-processed z=0 ROCKSTAR host halo catalog
		# eventually this step will not require a "pre-processed" halo catalog, but this if fine for now.

		if not isinstance(simulation_data['halos'],astropy.table.table.Table):
			raise TypeError("HOD_mock object requires an astropy Table halo catalog as input")
		table_of_halos = simulation_data['halos']

		if not isinstance(simulation_data['simulation_dict'],dict):
			raise TypeError("HOD_mock object requires a dictionary of simulation metadata as input")
		self.simulation_dict = simulation_data['simulation_dict']

		if not isinstance(hod_model,ho.HOD_Model):
			raise TypeError("HOD_mock object requires input hod_model to be an instance of halo_occupation.HOD_Model, or one of its subclasses")
		self.hod_model = hod_model

		if not isinstance(quenching_model,ho.Quenching_Model):
			raise TypeError("HOD_mock object requires input quenching_model to be an instance of halo_occupation.Quenching_Model, or one of its subclasses")
		self.quenching_model = quenching_model



		self.logM = np.array(np.log10(table_of_halos['MVIR']))
		self.haloID = np.array(table_of_halos['ID'])
		self.concen = ho.anatoly_concentration(self.logM)
		self.Rvir = np.array(table_of_halos['RVIR'])/1000.

		self.halopos = np.empty((len(table_of_halos),3),'f8')
		self.halopos.T[0] = np.array(table_of_halos['POS'][:,0])
		self.halopos.T[1] = np.array(table_of_halos['POS'][:,1])
		self.halopos.T[2] = np.array(table_of_halos['POS'][:,2])

		self.Lbox = self.simulation_dict['Lbox']

		if seed != None:
			np.random.seed(seed)

		self.idx_current_halo = 0 # index for current halo (bookkeeping device to speed up array access)


		#Set up the grid used to tabulate NFW profiles
		#This will be used to assign halo-centric distances to the satellites
		Npts_concen = defaults.default_Npts_concentration_array
		concentration_array = np.linspace(self.concen.min(),self.concen.max(),Npts_concen)
		Npts_radius = defaults.default_Npts_radius_array		
		radius_array = np.linspace(0.,1.,Npts_radius)
		
		self.cumulative_nfw_PDF = []
		# After executing the following lines, 
		#self.cumulative_nfw_PDF will be a list of functions. 
		#The list elements correspond to functions governing 
		#radial profiles of halos with different NFW concentrations.
		#Each function takes a scalar y in [0,1] as input, 
		#and outputs the x = r/Rvir corresponding to Prob_NFW( x > r/Rvir ) = y. 
		#Thus each list element is the inverse of the NFW cumulative NFW PDF.
		for c in concentration_array:
			self.cumulative_nfw_PDF.append(interp1d(ho.cumulative_NFW_PDF(radius_array,c),radius_array))

		#interp_idx_concen is an integer array with one element per host halo
		#each element gives the index pointing to the bins defined by concentration_array
		self.interp_idx_concen = np.digitize(self.concen,concentration_array)


	def _setup(self):
		"""
		Compute NCen,Nsat and preallocate arrays for __call__().
		
		"""

		self.NCen = num_cen_monte_carlo(self.logM,self.hod_model)
		self.hasCentral = self.NCen > 0

		self.NSat = np.zeros(len(self.logM),dtype=int)
		self.NSat[self.hasCentral] = num_sat_monte_carlo(
			self.logM[self.hasCentral],
			self.hod_model,output=self.NSat[self.hasCentral])

		self.would_have_quenched_central = quenched_monte_carlo(
			self.logM,self.quenching_model,'central')

		self.num_total_cens = self.NCen.sum()
		self.num_total_sats = self.NSat.sum()
		self.num_total_gals = self.num_total_cens + self.num_total_sats

		# preallocate output arrays
		self.coords = np.empty((self.num_total_gals,3),dtype='f8')
		self.logMhost = np.empty(self.num_total_gals,dtype='f8')
		self.isSat = np.zeros(self.num_total_gals,dtype='i4')
		self.isQuenched = np.zeros(self.num_total_gals,dtype='i4')
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

	def assign_satellite_positions(self,Nsat,center,r_vir,r_of_M,counter):
		""" 
		Note that I deleted John's M_halo argument since this does not appear to be used.

		"""

		satellite_system_coordinates = self.coords[counter:counter+Nsat,:]
		# Generate Nsat randoms in the interval [0,1)
		# finding the r_vir associated with that probability by inverting mass(r,r_vir,c)
		randoms = np.random.random(Nsat)
		r_random = r_of_M(randoms)*r_vir
		satellite_system_coordinates[:Nsat,:] *= r_random.reshape(Nsat,1)
		satellite_system_coordinates[:Nsat,:] += center.reshape(1,3)


	def populate(self,isSetup=False):
		"""
		Return a list of galaxies placed randomly in the halos.
		Returns coordinates, halo mass, isSat (boolean array with True for satellites)
		If isSetup is True, don't call _setup first (useful for calling from a child class).

		"""

		if not isinstance(self.hod_model,ho.HOD_Model):
			raise TypeError("HOD_mock object requires input hod_model to be an instance of halo_occupation.HOD_Model, or one of its subclasses")

		if not isinstance(self.quenching_model,ho.Quenching_Model):
			raise TypeError("HOD_mock object requires input quenching_model to be an instance of halo_occupation.Quenching_Model, or one of its subclasses")

		# pregenerate the output arrays
		if not isSetup:
			self._setup()

		# Preallocate centrals so we don't have to touch them again.
		self.coords[:self.num_total_cens] = self.halopos[self.hasCentral]
		self.logMhost[:self.num_total_cens] = self.logM[self.hasCentral]
		self.isQuenched[:self.num_total_cens] = self.would_have_quenched_central[self.hasCentral]


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
			logM,center,Nsat,r_vir,concen_idx = logmasses[self.ii],self.halopos[self.ii],self.NSat[self.ii],self.Rvir[self.ii],self.interp_idx_concen[self.ii]
			self.logMhost[counter:counter+Nsat] = logM
			self.assign_satellite_positions(Nsat,center,r_vir,self.cumulative_nfw_PDF[concen_idx-1],counter)
			counter += Nsat

		self.coords = enforce_periodicity_of_box(self.coords,self.Lbox)
		self.isQuenched[self.num_total_cens:-1] = quenched_monte_carlo(
			self.logMhost[self.num_total_cens:-1],
			self.quenching_model,'satellite')



	#...



























