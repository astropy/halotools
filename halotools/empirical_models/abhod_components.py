# -*- coding: utf-8 -*-
"""
Attempt to develop simple HOD models that include assembly bias.
"""

import numpy as np

# some dummy HOD for testing
def compute_ncen_mean(haloMasses,mmin,sigmalogM):
    """
    This provides a dummy HOD for testing purposes.
    """
    num_centrals = ( 1.0 + np.tanh((np.log10(haloMasses) - np.log10(mmin))/sigmalogM) )/2.0
    return num_centrals




# A class to implement 2-population, heaviside assembly bias
class HeavisideCentralAssemBiasModel:
    """
    This defines a class to deal with the assembly bias piece of the HOD, if present.
    """

    def __init__(self,percentile=0.50,dN=0.0):
        # establish a dictionary of model parameters and values
        self.abparameters=dict()
        self.abparameters['percentile']=100.0*percentile
        self.abparameters['deltaNupper']=dN
        self.abparameters['deltaNlower']=-percentile*dN/(1.0-percentile)
        # check that these parameter values do not violate number conservation
        self.check_ab_parameters()


    def check_ab_parameters(self):
        """
        Checks if the assembly bias parameters are valid. Right now, this does nothing
        because this requires some knowledge of how the halo information is being accessed
        by this class, and this is now unknown. This is critical here because these conditions
        must be met at all halo masses, so we need the entire catalog here.
        """
        pass  


    def compute_deltaN_2pop(self,haloXes):
        """
        This computes deltaN for a central population assuming our two-population
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











