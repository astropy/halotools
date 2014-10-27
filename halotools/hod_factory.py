# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
composite HOD models from a set of components. 

"""

import numpy as np

class HOD_Model(object):
    """ The most basic HOD model object. 
    The only methods are for assigning the mean occupation of a galaxy population 
    to a halo, the intra-halo radial profile of that population, and the 
    accompanying methods to generate Monte Carlo realizations of those methods. 

    All behavior is derived from external classes passed to the constructor via 
    component_model_dict, which serves as a set of instructions for how the 
    composite model is to be built from the components.  

    """

    def __init__(self,component_model_dict):
        """ The methods of this class derives their behavior from other, external classes, 
        passed in the form of the component_model_dict, a dictionary whose keys 
        are the galaxy types found in the halos, e.g., 'centrals', 'satellites', 'orphans', etc.
        The values of the component_model_dict are themselves dictionaries whose keys are 
        strings specifying the type of model being passes, e.g., 'occupation_model', and values 
        are instances of that type of model. The component_model_dict dictionary is built by 
        the hod_designer interface. 
        """
        self.component_model_dict = component_model_dict
        self.gal_types = self.component_model_dict.keys()

        self.parameter_dict = self.retrieve_all_inherited_parameters(self.component_model_dict)
        self.publications = []


    def mean_occupation(self,gal_type,*args):
        """ Method supplying the mean abundance of gal_type galaxies. 
        The behavior of this method is inherited from one of the component models. 
        """

        self.test_component_consistency(gal_type,'occupation_model')

        # For galaxies of type gal_type, the behavior of this method 
        # will be set by the inherited occupation_model object 
        occupation_model = self.component_model_dict[gal_type]['occupation_model']
        inherited_method = occupation_model.mean_occupation
        output_occupation = self.retrieve_inherited_behavior(inherited_method,args)

        return output_occupation


    def mc_occupation(self,gal_type,*args):
        """ Method providing a Monte Carlo realization of the mean occupation.
        The behavior of this method is inherited from one of the component models.
        """

        self.test_component_consistency(gal_type,'occupation_model')

        occupation_model = self.component_model_dict[gal_type]['occupation_model']
        inherited_method = occupation_model.mc_occupation
        output_mc_realization = self.retrieve_inherited_behavior(inherited_method,args)

        return output_mc_realization
        

    def mean_profile_parameters(self,gal_type,*args):
        """ Method returning the mean value of the parameters governing the radial profile 
        of gal_type galaxies. 
        The behavior of this method is inherited from one of the component models.
        """

        self.test_component_consistency(gal_type,'profile_model')

        profile_model = self.component_model_dict[gal_type]['profile_model']
        inherited_method = occupation_model.mean_profile_parameters
        output_profiles = self.retrieve_inherited_behavior(inherited_method,args)

        return output_profiles

    def mc_profile(self,gal_type,*args):
        """ Method returning a Monte Carlo realization of the radial profile. 
        The behavior of this method is inherited from one of the component models.
        """

        self.test_component_consistency(gal_type,'profile_model')

        profile_model = self.component_model_dict[gal_type]['profile_model']
        inherited_method = occupation_model.mc_profile
        output_mc_realization = self.retrieve_inherited_behavior(inherited_method,args)

        return output_mc_realization

    def test_component_consistency(self,gal_type,component_key):
        """ Simple tests to run to make sure that the desired behavior 
        can be found in the component model.
        """

        if gal_type not in self.gal_types:
            raise KeyError("Input gal_type is not supported "
                "by any of the components of this composite model")         

        if component_key not in self.component_model_dict[gal_type]:
            raise KeyError("Could not find method to retrieve "
                " inherited behavior from the provided component model")

    def retrieve_inherited_behavior(self,inherited_method,*args):
        """ Method whose function is solely to call the component model methods 
        using the correct number of arguments. 
        """

        if len(args)==1:
            primary_haloprop = args[0]
            output = inherited_method(primary_haloprop)
        elif len(args)==2:
            primary_haloprop, secondary_haloprop = args[0], args[1]
            output = inherited_method(primary_haloprop,secondary_haloprop)
        else:
            raise TypeError("Only one or two halo property inputs are supported by "
                "mean_occupation method")

        return output

    def retrieve_all_inherited_parameters(self,component_model_dict):

        output_dict = {}

        # Loop over all galaxy types in the composite model
        for gal_type, model_list in component_model_dict.iteritems():
            # For each galaxy type, loop over its features
            for model_feature in model_list:
                # Check to make sure we're not duplicating any dictionary keys
                self.test_model_redundancy(
                    output_dict,model_feature.parameter_dict)
                output_dict = dict(
                    model_feature.parameter_dict.items() + 
                    output_dict.items())

        return output_dict

    def test_model_redundancy(self,existing_composite_model,new_model_component):
        """ Check whether the new_model_component dictionary contains 
        keys that duplicate the keys in the existing_composite_model dictionary.

        """

        intersection = list(set(existing_composite_model) & set(new_model_component))
        print(set(existing_composite_model),set(new_model_component))
        if intersection != []:
            raise KeyError("New component model contains duplicate parameter keys")
















