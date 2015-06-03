# -*- coding: utf-8 -*-
"""
Attempt to develop simple HOD models that include assembly bias.
"""

import numpy as np # numpy as usual
import hod_components # Andrew Hearin's HOD components
import model_defaults # default parameters for HOD models

# some dummy HOD for testing
def compute_ncen_mean(haloMasses,mmin,sigmalogM):
    """
    This provides a dummy HOD for testing purposes.
    """
    num_centrals = ( 1.0 + np.tanh((np.log10(haloMasses) - np.log10(mmin))/sigmalogM) )/2.0
    return num_centrals




# A class to implement 2-population, heaviside assembly bias
class HeavisideCenAssemBiasModel(hod_components.OccupationComponent):
    """
    This defines a class to deal with the assembly bias piece of the HOD, if present.
    In this iteration, I am attempting to sub-class it to OccupationComponent.
    """

    def __init__(self,
        standard_cen_model,
        percentile=0.50,frac_dNmax=0.0,
        input_param_dict=None,
        haloprop_key_dict=model_defaults.haloprop_key_dict,
        threshold=model_defaults.default_luminosity_threshold,
        sec_haloprop_key='vmax'
        gal_type='centrals'):
        
        """
        argument: standard_cen_model is an instance of a class that can compute 
        standard central occupations. we require its mean_occupation method. 
        """
        
        # establish the non-assembly biased model that these routines will
        # be working with.

        self.standard_cen_model=standard_cen_model

        # establish a dictionary of model parameters and values
        self.abparameters=dict()
        self.abparameters['percentile']=100.0*percentile
        self.abparameters['frac_dNmax']=frac_dNmax

        # call super-class init routine
        occupation_bound=1.0
        hod_components.OccupationComponent.__init__(self,gal_type,
            haloprop_key_dict,threshold,occupation_bound)

        # set parameter dictionary
        self._set_param_dict()

        # inherit the standard model's mean occupation routine
        self.mean_occupation = standard_cen_model.mean_occupation

        # check that these parameter values do not violate number conservation
        self.check_valid_ab_parameters()

        # assign halos percentiles based on the secondary property
        self.assign_halo_secondary_percentiles()


    # checks validity of the input parameters
    def check_valid_ab_parameters(self):
        """
        Checks if the assembly bias parameters are valid. Right now, this does nothing
        because this requires some knowledge of how the halo information is being accessed
        by this class, and this is now unknown. This is critical here because these conditions
        must be met at all halo masses, so we need the entire catalog here.
        """
        pass

    def _set_param_dict(self):
        """
        Builds parameter dictionary.
        """
        self.param_dict=dict()


    # assign halos percentile values of the secondary halo property
    def assign_halo_secondary_percentiles(self):
        """
        Takes the current halo catalog and, within mass bins, orders the 
        halos according to their secondary property.
        """
        pass

    # routine to compute the shift in the mean occupation due to assembly bias
    def compute_deltaN_2pop(self,haloXes):
        """
        This computes deltaN for a central population assuming two-population
        Heavside assembly bias based on the property haloXes.

        This version of the routine assumes population of halos with a single mass.

        This also assumes that the parameters have been checked for validity.
        """
        pivot=np.percentile(haloXes,self.abparameters['percentile'])
        
        # I feel as though there must be a better way to do this
        deltahi=np.ones(np.shape(haloXes))*self.abparameters['deltaNupper']
        deltalo=np.ones(np.shape(haloXes))*self.abparameters['deltaNlower']
        
        deltaN=np.where(haloXes>=pivot,deltahi,deltalo)
        
        return deltaN

    # compute mean halo occupation
    def mean_occupation(self):
        """
        Method to compute mean halo occupation of halos in the halo catalog.
        """
        pass


    def mc_occupation(self):
        """
        Method to compute Monte Carlo realization of the occupation model.
        """
        pass

    # routine to compute non-assembly biased mean occupation
    def standard_mean_occupation(self,*args):
        """
        Compute the mean occupation of halos WITHOUT assembly bias. 
        This will use the standard model instance that this is instantiated 
        with.
        """
        return self.standard_cen_model.mean_occupation(*args)









