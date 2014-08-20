
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)
""" The make_mocks module contains the classes and functions used 
to populate N-body simulations with realizations of galaxy-halo models. 
The functions in the halo_occupation module are used to define the analytical models
of both stellar mass and quenching; this module paints monte carlo realizations of 
those models onto halos in an N-body catalog at a single snapshot.
Currently only set up for HOD-type models, but near-term features include 
CLF/CSMF models, and (conditional) abundance matching models.
Class design is built around future MCMC applications, so that 
lower level objects like numpy ndarrays are used to store object attributes, 
for which it is cheaper and faster to allocate memory."""

__all__= ['enforce_periodicity_of_box','quenched_monte_carlo','HOD_mock']

import read_nbody
import halo_occupation as ho
import numpy as np

from scipy.interpolate import interp1d as interp1d
from scipy.stats import poisson

import defaults
import os
from copy import copy
from collections import Counter
import astropy
import time


def enforce_periodicity_of_box(coords, box_length):
    """ Function used to apply periodic boundary conditions 
    of the simulation, so that mock galaxies all lie in the range [0, Lbox].

    Parameters
    ----------
    coords : array_like
        description of coords
        
    box_length : scalar
        the size of simulation box (currently hard-coded to be Mpc/h units)

    Returns
    -------
    coords : array_like
        1d list of floats giving coordinates that have been corrected for box periodicity

    """
    test = coords > box_length
    coords[test] = coords[test] - box_length
    test = coords < 0
    coords[test] = box_length + coords[test]
    return coords


class HOD_mock(object):
    """     

    Parameters
    ----------
    simulation_data : optional
        simulation_data is an instance of the `simulation` class 
        defined in the `read_nbody` module. 
        Currently only Bolshoi at z=0 is supported.

    halo_occupation_model : optional 
        halo_occupation_model is an instance of the 
        `HOD_Model` class defined in the `halo_occupation` module.

    seed : float, optional
        Random number seed. Currently unused. Will be useful when implementing an MCMC.

    """

    def __init__(self,simulation_data=None,
        halo_occupation_model=ho.vdB03_Quenching_Model,threshold = -20,seed=None):

        # If no simulation object is passed to the constructor, 
        # the default simulation will be chosen
        # Currently this set to be Bolshoi at z=0, 
        # as specified in the defaults module
        if simulation_data is None:
            simulation_data = read_nbody.simulation()

        # Test to make sure the simulation data is the appropriate type
        if not isinstance(simulation_data.halos,astropy.table.table.Table):
            raise TypeError("HOD_mock object requires an astropy Table object as input")

        # Bind halo catalog to the HOD_mock object
        self.halos = simulation_data.halos
        self.Lbox = simulation_data.Lbox

        # Add columns to the halos table attribute of the mock object
        self.halos['HALO_TYPE_CENTRALS']=np.ones(len(self.halos))
        self.halos['HALO_TYPE_SATELLITES']=np.ones(len(self.halos))
        self.halos['PRIMARY_HALO_PROPERTY']=np.zeros(len(self.halos))
        self.halos['SECONDARY_HALO_PROPERTY']=np.zeros(len(self.halos))

        # Test to make sure the hod model is the appropriate type
        hod_model_instance = halo_occupation_model(threshold=threshold)
        if not isinstance(hod_model_instance,ho.HOD_Model):
            raise TypeError("HOD_mock object requires input halo_occupation_model "
                "to be an instance of halo_occupation.HOD_Model, or one of its subclasses")
        # Bind the instance of the hod model to the HOD_mock object
        self.halo_occupation_model = hod_model_instance

        # Create numpy arrays containing data from the halo catalog and bind them to the mock object
        self.primary_halo_property = np.array(
            self.halos[self.halo_occupation_model.primary_halo_property_key])
        self.halos['PRIMARY_HALO_PROPERTY']=np.array(
            self.halos[self.halo_occupation_model.primary_halo_property_key])

        if self.halo_occupation_model.primary_halo_property_key == 'MVIR':
            self.primary_halo_property = np.log10(self.primary_halo_property)
            self.halos['PRIMARY_HALO_PROPERTY']=np.log10(self.halos['PRIMARY_HALO_PROPERTY'])

        if isinstance(self.halo_occupation_model,ho.Assembly_Biased_HOD_Model):
            self.secondary_halo_property = np.array(
                self.halos[self.halo_occupation_model.secondary_halo_property_key])
            self.halos['SECONDARY_HALO_PROPERTY'] = np.array(
                self.halos[self.halo_occupation_model.secondary_halo_property_key])

        self.halo_type = self.halos['HALO_TYPE_CENTRALS']


        self.haloID = np.array(self.halos['ID'])
        self.concen = self.halo_occupation_model.mean_concentration(
            self.halos['PRIMARY_HALO_PROPERTY'],self.halos['HALO_TYPE_CENTRALS'])

        self.Rvir = np.array(self.halos['RVIR'])/1000.

        self.halopos = np.empty((len(self.halos),3),'f8')
        self.halopos.T[0] = np.array(self.halos['POS'][:,0])
        self.halopos.T[1] = np.array(self.halos['POS'][:,1])
        self.halopos.T[2] = np.array(self.halos['POS'][:,2])

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
        Compute NCen,Nsat and preallocate various arrays.
        No inputs; returns nothing; only modifies attributes 
        of the HOD_mock object to which the method is bound.

        Warning
        -------
        The basic behavior of this method will soon be changed, correspondingly 
        changing the basic API of the mock making.
        
        """

        #if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
        #    self.self.halos['HALO_TYPE_CENTRALS'] = self.quenched_monte_carlo(
        #        self.primary_halo_property,self.halo_occupation_model,'central')

        self.NCen = self.num_cen_monte_carlo(
            self.primary_halo_property,self.halo_type)
        self.hasCentral = self.NCen > 0

        self.NSat = np.zeros(len(self.primary_halo_property),dtype=int)
        self.NSat[self.hasCentral] = self.num_sat_monte_carlo(
            self.primary_halo_property[self.hasCentral],
            self.halo_type[self.hasCentral],
            output=self.NSat[self.hasCentral])

        self.num_total_cens = self.NCen.sum()
        self.num_total_sats = self.NSat.sum()
        self.num_total_gals = self.num_total_cens + self.num_total_sats

        # preallocate output arrays
        self.coords = np.empty((self.num_total_gals,3),dtype='f8')
        self.logMhost = np.empty(self.num_total_gals,dtype='f8')
        self.isSat = np.zeros(self.num_total_gals,dtype='i4')
        #if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
        #    self.isQuenched = np.zeros(self.num_total_gals,dtype='i4')
    #...

    def _random_angles(self,coords,start,end,N):
        """
        Generate a list of random angles. 
        Assign the angles to coords[start:end], 
        an index bookkeeping trick to speed up satellite position assignment.

        Notes 
        -----
        API is going to change, so that function actually returns values, 
        rather than privately over-writing object attributes. 
        
        """

        cos_t = np.random.uniform(-1.,1.,N)
        phi = np.random.uniform(0,2*np.pi,N)
        sin_t = np.sqrt((1.-cos_t**2))
        
        coords[start:end,0] = sin_t * np.cos(phi)
        coords[start:end,1] = sin_t * np.sin(phi)
        coords[start:end,2] = cos_t
    #...

    def assign_satellite_positions(self,Nsat,center,r_vir,r_of_M,counter):
        """ Use pre-tabulated cumulative_NFW_PDF to 
        draw random satellite positions. 

        Parameters
        ----------
        Nsat : Number of satellites in the host system whose positions are being assigned.
        
        center : array_like
        position of the halo hosting the satellite system whose positions are being assigned.

        r_vir : array_like
        Virial radius of the halo hosting the satellite system whose positions are being assigned.

        r_of_M : 
            Function object defined by scipy interpolation of cumulative_NFW_PDF.

        counter : int
            bookkeeping device controlling indices of satellites in the host system whose positions are being assigned.

        Notes 
        -----
        API is going to change, so that function actually returns values, 
        rather than privately over-writing object attributes. 

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
        Assign positions to mock galaxies. 
        Returns coordinates, halo mass, isSat (boolean array with True for satellites)
        If isSetup is True, don't call _setup first (useful for future MCMC applications).

        """

        if not isinstance(self.halo_occupation_model,ho.HOD_Model):
            raise TypeError("HOD_mock object requires input hod_model "
                "to be an instance of halo_occupation.HOD_Model, or one of its subclasses")

        # pregenerate the output arrays
        if not isSetup:
            self._setup()

        # Preallocate centrals so we don't have to touch them again.
        self.coords[:self.num_total_cens] = self.halopos[self.hasCentral]
        self.logMhost[:self.num_total_cens] = self.primary_halo_property[self.hasCentral]

        #if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
        #    self.isQuenched[:self.num_total_cens] = self.halo_isQuenched[self.hasCentral]


        counter = self.num_total_cens
        self.idx_current_halo = 0

        self.isSat[counter:] = 1 # everything else is a satellite.
        # Pregenerate satellite angles all in one fell swoop
        self._random_angles(self.coords,counter,self.coords.shape[0],self.num_total_sats)

        # all the satellites will now end up at the end of the array.
        satellite_index_array = np.nonzero(self.NSat > 0)[0]
        # these two save a bit of time by eliminating calls to records.__getattribute__
        logmasses = self.primary_halo_property


        # The following loop assigning satellite positions takes up nearly 100% of the mock population time
        start = time.time()
        for self.ii in satellite_index_array:
            logM = logmasses[self.ii]
            center = self.halopos[self.ii]
            Nsat = self.NSat[self.ii]
            r_vir = self.Rvir[self.ii]
            concen_idx = self.interp_idx_concen[self.ii]
            self.logMhost[counter:counter+Nsat] = logM

            self.assign_satellite_positions(Nsat,center,r_vir,self.cumulative_nfw_PDF[concen_idx-1],counter)
            counter += Nsat
        runtime = time.time() - start
        #print(str(runtime)+' seconds to assign satellite positions')

        self.coords = enforce_periodicity_of_box(self.coords,self.Lbox)

        #if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
        #    self.isQuenched[self.num_total_cens:-1] = self.quenched_monte_carlo(
        #        self.logMhost[self.num_total_cens:-1],
        #        self.halo_occupation_model,'satellite')

    def num_cen_monte_carlo(self,primary_halo_property,halo_type):
        """ Returns Monte Carlo-generated array of 0 or 1 specifying whether there is a central in the halo.

        Parameters
        ----------
        logM : float or array

        hod_model : 
            HOD_Model object defined in halo_occupation module.

        Returns
        -------
        num_ncen_array : int or array

        
        """
        mean_ncen_array = self.halo_occupation_model.mean_ncen(
            primary_halo_property,halo_type)

        num_ncen_array = np.array(
            mean_ncen_array > np.random.random(len(primary_halo_property)),dtype=int)

        return num_ncen_array

    def num_sat_monte_carlo(self,primary_halo_property,halo_type,output=None):
        """  Returns Monte Carlo-generated array of integers specifying the number of satellites in the halo.

        Parameters
        ----------
        logM : float or array

        hod_model : HOD_Model object defined in halo_occupation module.

        Returns
        -------
        num_nsat_array : int or array
            Values of array specify the number of satellites hosted by each halo.


        """
        Prob_sat = self.halo_occupation_model.mean_nsat(
            primary_halo_property,halo_type)
        # NOTE: need to cut at zero, otherwise poisson bails
        # BUG IN SCIPY: poisson.rvs bails if there are zeroes in a numpy array
        test = Prob_sat <= 0
        Prob_sat[test] = defaults.default_tiny_poisson_fluctuation

        num_nsat_array = poisson.rvs(Prob_sat)

        return num_nsat_array

    def quenched_monte_carlo(self,primary_halo_property,halo_type,galaxy_type):
        """ Returns Monte Carlo-generated array of 0 or 1 specifying whether the galaxy is quenched.
        Parameters
        ----------
        logM : array_like

        halo_occupation_model : 
            Any HOD_Quenching_Model object defined in halo_occupation module.

        galaxy_type : string
            Only supported values are 'central' or 'satellite'. Used to indicate which 
            quenching model method should used to generate the Monte Carlo.

            Returns
            -------
            quenched_array : int or array
                Used to define whether mock galaxy is quenched (1) or star-forming (0)

        """

        if not isinstance(self.halo_occupation_model,ho.HOD_Quenching_Model):
            raise TypeError("input halo_occupation_model must be an instance of a supported HOD_Quenching_Model, or one if its derived subclasses")

        if galaxy_type == 'central':
            quenching_function = self.halo_occupation_model.mean_quenched_fraction_centrals
        elif galaxy_type == 'satellite':
            quenching_function = self.halo_occupation_model.mean_quenched_fraction_satellites
        else:
            raise TypeError("input galaxy_type must be a string set either to 'central' or 'satellite'")


        quenched_array = np.array(
            quenching_function(primary_halo_property,halo_type) > np.random.random(
                len(primary_halo_property)),dtype=int)

        return quenched_array


    #...



























