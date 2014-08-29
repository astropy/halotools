
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)
""" This module contains the classes and functions used 
to populate N-body simulations with realizations of galaxy-halo models. 
Currently only set up for HOD-style models, but near-term features include 
CLF/CSMF models, and (conditional) abundance matching models.
Class design is built around future MCMC applications. """

__all__= ['enforce_periodicity_of_box','HOD_mock']

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
import warnings 
from astropy.table import Table

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
    """ Class used to build mock realizations of any HOD-style model defined in `halo_occupation` module.

    Instances of this class represent a mock galaxy distribution whose properties 
    depend on the style of HOD model passed to the constructor, and on the 
    parameter values of the model. 

    Currently supported models are `~halotools.halo_occupation.Zheng07_HOD_Model`, 
    `~halotools.halo_occupation.Satcen_Correlation_Polynomial_HOD_Model`, 
    `~halotools.halo_occupation.Polynomial_Assembias_HOD_Model`, 
    and `~halotools.halo_occupation.vdB03_Quenching_Model`.

    To create a mock, instantiate this class and then call the `populate` method.

    Parameters
    ----------
    simulation_data : optional
        simulation_data is an instance of the `~halotools.read_nbody.simulation` class 
        defined in the `~halotools.read_nbody` module. 

    halo_occupation_model : optional 
        halo_occupation_model is any subclass of the abstract class 
        `~halotools.halo_occupation.HOD_Model` defined 
        in the `~halotools.halo_occupation` module. 

    threshold : optional
        Luminosity or stellar mass threshold of the mock galaxy sample.

    seed : float, optional
        Random number seed. Currently ignored. Will be useful when implementing an MCMC.

    tableBundle : boolean, optional
        If set to True, the class instance will have a `galaxies` attribute, 
        which is an astropy Table providing a convenient bundle of the mock.

    """

    def __init__(self,simulation_data=None,
        halo_occupation_model=ho.vdB03_Quenching_Model,threshold = -20,
        seed=None,tableBundle=True):

        # If no simulation_data object is passed to the constructor, 
        # the default simulation will be chosen
        # Currently this set to be Bolshoi at z=0, 
        # as specified in the defaults module
        if simulation_data is None:

            simulation_name=defaults.default_simulation_name
            scale_factor=defaults.default_scale_factor
            halo_finder=defaults.default_halo_finder

            simulation_data = read_nbody.simulation(
                simulation_name,scale_factor,halo_finder)

        # Test to make sure the simulation data is the appropriate type
        if not isinstance(simulation_data.halos,astropy.table.table.Table):
            raise TypeError("HOD_mock object requires an astropy Table object as input")


        # Test to make sure the hod model is the appropriate type
        hod_model_instance = halo_occupation_model(threshold=threshold)
        if not isinstance(hod_model_instance,ho.HOD_Model):
            raise TypeError("HOD_mock object requires input halo_occupation_model "
                "to be an instance of halo_occupation.HOD_Model, or one of its subclasses")
        # Bind the instance of the hod model to the HOD_mock object
        self.halo_occupation_model = hod_model_instance

        # Bind halo catalog to the HOD_mock object
        self.halos = simulation_data.halos
        self.Lbox = simulation_data.Lbox


        # Create numpy ndarrays containing data from the halo catalog and bind them to the mock object
        self._primary_halo_property = np.array(
            self.halos[self.halo_occupation_model.primary_halo_property_key])
        # Use log10Mvir instead of Mvir if this is the primary halo property
        if self.halo_occupation_model.primary_halo_property_key == 'MVIR':
            self._primary_halo_property = np.log10(self._primary_halo_property)
            #self.halos['PRIMARY_HALO_PROPERTY']=np.log10(self.halos['PRIMARY_HALO_PROPERTY'])

        self._halo_type_centrals = np.ones(len(self._primary_halo_property))
        self._halo_type_satellites = np.ones(len(self._primary_halo_property))

        # If the mock was passed an assembly-biased HOD model, 
        # set the secondary halo property and compute halo_types 
        if isinstance(self.halo_occupation_model,ho.Assembias_HOD_Model):

            # If assembly bias is desired for centrals, implement it.
            if self.halo_occupation_model.secondary_halo_property_centrals_key != None:
                #self.halos['SECONDARY_HALO_PROPERTY_CENTRALS'] = np.array(
                    #self.halos[self.halo_occupation_model.secondary_halo_property_centrals_key])
                self._secondary_halo_property_centrals = (
                    self.halos[self.halo_occupation_model.secondary_halo_property_centrals_key])
                self._halo_type_centrals = (
                    self.halo_occupation_model.halo_type_calculator(
                    self._primary_halo_property,
                    self._secondary_halo_property_centrals,
                    self.halo_occupation_model.halo_type1_fraction_centrals))

            # If assembly bias is desired for satellites, implement it.
            if self.halo_occupation_model.secondary_halo_property_satellites_key != None: 

                self._secondary_halo_property_satellites = np.array(
                    self.halos[self.halo_occupation_model.secondary_halo_property_satellites_key])
                self._halo_type_satellites = (
                    self.halo_occupation_model.halo_type_calculator(
                    self._primary_halo_property,
                    self._secondary_halo_property_satellites,
                    self.halo_occupation_model.halo_type1_fraction_satellites))

        # If the model includes quenching designations, pre-allocate an array 
        # dedicated to whether or not the central galaxy that would be in a halo 
        # will be quenched. Doing this in advance costs nothing, and simplifies 
        # the unification of models with a distinct quenched/active SMHM 
        self._quenched_halo = np.zeros(len(self._primary_halo_property))

        self._concen = self.halo_occupation_model.mean_concentration(
            self._primary_halo_property,self._halo_type_centrals)

        self._rvir = np.array(self.halos['RVIR'])/1000.

        self._halopos = np.empty((len(self.halos),3),'f8')
        self._halopos.T[0] = np.array(self.halos['POS'][:,0])
        self._halopos.T[1] = np.array(self.halos['POS'][:,1])
        self._halopos.T[2] = np.array(self.halos['POS'][:,2])

        if seed != None:
            np.random.seed(seed)

        #self.idx_current_halo = 0 # index for current halo (bookkeeping device to speed up array access)

        #Set up the grid used to tabulate NFW profiles
        #This will be used to assign halo-centric distances to the satellites
        Npts_concen = defaults.default_Npts_concentration_array
        concentration_array = np.linspace(self._concen.min(),self._concen.max(),Npts_concen)
        Npts_radius = defaults.default_Npts_radius_array        
        radius_array = np.linspace(0.,1.,Npts_radius)
        
        self._cumulative_nfw_PDF = []
        # After executing the following lines, 
        # self._cumulative_nfw_PDF will be a  (private) list of functions. 
        # The elements of this list are functions governing 
        # radial profiles of halos with different NFW concentrations.
        # Each function takes a scalar y in [0,1] as input, 
        # and outputs the x = r/Rvir corresponding to Prob_NFW( x < r/Rvir ) = y. 
        # Thus each list element is the inverse of the NFW cumulative PDF.
        for c in concentration_array:
            self._cumulative_nfw_PDF.append(interp1d(ho.cumulative_NFW_PDF(radius_array,c),radius_array))

        #interp_idx_concen is a (private) integer array with one element per host halo
        #each element gives the index pointing to the bins defined by concentration_array
        self._interp_idx_concen = np.digitize(self._concen,concentration_array)

        self.tableBundle = tableBundle


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

        # If the HOD Model passed to the constructor has features supporting 
        # quenched/star-forming designations, use the method 
        # self.halo_occupation_model.mean_quenched_fraction_centrals
        # to assign quenched/star-forming designations to the HALOS.
        # Note that there will thus be many halos with a quenched designation 
        # but yet without a central. While this may seem strange, 
        # this needs to be done at this step, rather than after assigning 
        # centrals to halos, to support models in which central galaxy 
        # occupation statistics explicitly  
        # depend on quenching/star-forming designation, 
        # such as Tinker, Leauthaud, et al. 2013.
        if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
            self._quenched_halo = (self.quenched_monte_carlo(
                self._primary_halo_property,
                self._halo_type_centrals,
                galaxy_type='central'))

        self._NCen = self.num_cen_monte_carlo(
            self._primary_halo_property,self._halo_type_centrals)
        self._hasCentral = self._NCen > 0

        # If implementing central-satellite correlations, 
        # set the satellite halo type according to whether the halo 
        # hosts a central.
        if isinstance(self.halo_occupation_model,ho.Satcen_Correlation_Polynomial_HOD_Model):
            self._halo_type_satellites = self._NCen

        self._NSat = np.zeros(len(self._primary_halo_property),dtype=int)

        self._NSat = self.num_sat_monte_carlo(
            self._primary_halo_property,
            self._halo_type_satellites,
            output=self._NSat)

        self.num_total_cens = self._NCen.sum()
        self.num_total_sats = self._NSat.sum()
        self.num_total_gals = self.num_total_cens + self.num_total_sats

        # preallocate output arrays
        self.coords = np.empty((self.num_total_gals,3),dtype='f8')
        self.coordshost = np.empty((self.num_total_gals,3),dtype='f8')
        self.logMhost = np.empty(self.num_total_gals,dtype='f8')
        self.isSat = np.zeros(self.num_total_gals,dtype='i4')
        self.halo_type = np.ones(self.num_total_gals,dtype='f8')

        if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
            self.isQuenched = np.zeros(self.num_total_gals,dtype='f8')

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

        # Assign properties to centrals. Note that as a result of this step, 
        # the first num_total_cens entries of the mock object ndarrays 
        # pertain to centrals. 
        self.coords[:self.num_total_cens] = self._halopos[self._hasCentral]
        self.coordshost[:self.num_total_cens] = self._halopos[self._hasCentral]
        self.logMhost[:self.num_total_cens] = self._primary_halo_property[self._hasCentral]
        self.halo_type[:self.num_total_cens] = self._halo_type_centrals[self._hasCentral]


        if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
            self.isQuenched[:self.num_total_cens] = (
                self._quenched_halo[self._hasCentral])


        counter = self.num_total_cens
        #self.idx_current_halo = 0

        self.isSat[counter:] = 1 # everything else is a satellite.
        # Pregenerate satellite angles all in one fell swoop
        self._random_angles(self.coords,counter,self.coords.shape[0],self.num_total_sats)

        # all the satellites will now end up at the end of the array.
        satellite_index_array = np.nonzero(self._NSat > 0)[0]
        # these two save a bit of time by eliminating calls to records.__getattribute__
        logmasses = self._primary_halo_property
        halo_type_satellites = self._halo_type_satellites


        # The following loop assigning satellite positions takes up nearly 100% of the mock population time
        start = time.time()
        for self.ii in satellite_index_array:
            logM = logmasses[self.ii]
            halo_type = halo_type_satellites[self.ii]
            center = self._halopos[self.ii]
            Nsat = self._NSat[self.ii]
            r_vir = self._rvir[self.ii]
            concen_idx = self._interp_idx_concen[self.ii]

            self.logMhost[counter:counter+Nsat] = logM
            self.halo_type[counter:counter+Nsat] = halo_type
            self.coordshost[counter:counter+Nsat] = center

            self.assign_satellite_positions(Nsat,center,r_vir,self._cumulative_nfw_PDF[concen_idx-1],counter)
            counter += Nsat
        runtime = time.time() - start
        #print(str(runtime)+' seconds to assign satellite positions')

        self.coords = enforce_periodicity_of_box(self.coords,self.Lbox)

        if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
            self.isQuenched[self.num_total_cens:-1] = self.quenched_monte_carlo(
                self.logMhost[self.num_total_cens:-1],
                self.halo_type[self.num_total_cens:-1],'satellite')

        if self.tableBundle==True:
            self.galaxies = self.table_bundle()

    def table_bundle(self):
        """ Create an astropy Table object and bind it to the mock object.

        """

        column_names = (['coords','coordshost',
            'primary_halo_property','halo_type',
            'isSat'])

        tbdata = ([self.coords,self.coordshost,self.logMhost,
            self.halo_type,self.isSat])
    
        if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
            column_names.append('isQuenched')
            tbdata.append(self.isQuenched)

        galaxy_table = Table(tbdata,names=column_names)

        return galaxy_table

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



























