# -*- coding: utf-8 -*-
"""

Update to the halo_occupation module that 
is instead built around the decorator desing pattern. 

"""
import numpy as np
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
import defaults

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

# There should also be a halo_type decorator.
# Both inflection and conformity classes need this knowledge.
# A halo catalog need not be assigned types by the model. 
# The model only needs to know the typing function.
# The mock factory will assign explicit types 
# to each halo in the halo catalog, during the build phase.

#@six.add_metaclass(ABCMeta)

def main():

    m = HOD_Model()
    central_params={'Mmin':11}
    m = Weinberg_Centrals(central_params,m)
    print('')
    print('Galaxy types: ',m.gal_types)
    print('model parameters:',m.parameter_dict)
    print('Mean Ncen:',m.mean_occupation('centrals'))
    #print('Mean Nsat:',m.mean_occupation('satellites'))
    print('')

    satellite_params={'alpha':1.02}
    m = Weinberg_Centrals(satellite_params,m)
    print('')
    print('Galaxy types: ',m.gal_types)
    print('model parameters:',m.parameter_dict)
    print('Mean Ncen:',m.mean_occupation('centrals'))
    #print('Mean Nsat:',m.mean_occupation('satellites'))
    print('')



    return


class HOD_Model(object):
    """ Base class for any HOD-style model of the galaxy-halo connection.

    """
    def __init__(self):
        self.parameter_dict = {}
        self.gal_types=[]
        self.publications = []
        self.threshold = None
        self.baseline_model = None

    def extend_params(self,new_params=None,new_gal_type=None):
        """ This should be written only once, to avoid copying and pasting.
        Still needs debugging."""
        
        # Extend the parameter dictionary
        if new_params is not None:
            self.test_model_redundancy(
                self.baseline_model.parameter_dict,new_params)
            self.parameter_dict = dict(
                self.baseline_model.parameter_dict.items() + 
                new_params.items())

        # Extend the gal_types
        if new_gal_type is not None:
            self.test_model_redundancy(self.baseline_model.gal_types,[new_gal_type])
            self.gal_types = self.baseline_model.gal_types
            self.gal_types.extend([new_gal_type])

    def test_model_redundancy(self,existing_model_attr,new_model_attr):

        intersection = list(set(existing_model_attr) & set(new_model_attr))
        print(set(existing_model_attr),set(new_model_attr))
        if intersection != []:
            raise TypeError("New Model contains redundant names")

    def mean_occupation(self,gal_type):
        raise TypeError("Mean Occupation for input gal_type has not been defined")


class Weinberg_Centrals(HOD_Model):
    """ Traditional erf model of central occupations. 
    """
    def __init__(self,params,baseline_model=None,
        extend_params=True):

        # If this is the first instance in the building of a decorated model, 
        # there will be no baseline_model passed ot the constructor. In this case, 
        # run the HOD_Model constructor to initialize the primary attributes
        # This sets parameter_dict, gal_types, and publiations to empty lists, 
        # and threshold and baseline_model to None
        if baseline_model is None:
            HOD_Model.__init__(self)
        self.baseline_model=baseline_model
        if extend_params is True:
            self.extend_params(params,'centrals')

    def mean_occupation(self,gal_type):
        if gal_type == 'centrals':
            return 1
        else:
            return self.baseline_model.mean_occupation(gal_type)
            

class Berlind_Satellites(HOD_Model):
    """ Traditional power law model of satellite occupations. 
    """
    def __init__(self,params,baseline_model,
        extend_params=True):

        # If this is the first instance in the building of a decorated model, 
        # there will be no baseline_model passed ot the constructor. In this case, 
        # run the HOD_Model constructor to initialize the primary attributes
        # This sets parameter_dict, gal_types, and publiations to empty lists, 
        # and threshold and baseline_model to None
        if baseline_model is None:
            HOD_Model.__init__(self)
        self.baseline_model=baseline_model
        if extend_params is True:
            self.extend_params(params,'satellites')

    def mean_occupation(self,gal_type):
        if gal_type == 'satellites':
            return 2
        else:
            return self.baseline_model.mean_occupation(gal_type)











###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
    main()






