# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
composite HOD models from a set of components. 

"""

__all__ = ['HodModel']

import numpy as np
import occupation_helpers as occuhelp

class HodModel(object):
    """ Composite HOD model object. 
    The primary methods are for assigning the mean occupation of a galaxy population 
    to a halo, the intra-halo radial profile of that population, and the 
    accompanying methods to generate Monte Carlo realizations of those methods. 

    All behavior is derived from external classes passed to the constructor via 
    component_model_dict, which serves as a set of instructions for how the 
    composite model is to be built from the components. 

    """

    def __init__(self, halo_prof_model, component_model_dict):
        """ The methods of this class derive their behavior from other, external classes, 
        passed in the form of the component_model_dict, a dictionary whose keys 
        are the galaxy types found in the halos, e.g., 'centrals', 'satellites', 'orphans', etc.
        The values of the component_model_dict are themselves dictionaries whose keys  
        specify the type of model being passed, e.g., 'occupation_model', and values 
        are instances of that type of model. The component_model_dict dictionary is built by 
        the hod_designer interface. The input halo_prof_model is an instance of the class 
        governing the assumed profile of the underlying halos. 

        """
        # Bind the model-building instructions to the composite model
        self.halo_prof_model = halo_prof_model
        self.component_model_dict = component_model_dict

        # Create attributes for galaxy types and their occupation bounds
        self.gal_types = self.component_model_dict.keys()
        self.occupation_bound = {}
        for gal_type in self.gal_types:
            self.occupation_bound[gal_type] = (
                self.component_model_dict[gal_type]['occupation_model'].occupation_bound)

        # Create strings used by the MC methods to access the appropriate columns of the 
        # halo table passed by the mock factory
        # Currently, all behaviors of all galaxy types must use 
        # the same primary (and, if applicable, secondary) halo property. 
        # Also create a dictionary for which gal_types, and which behaviors, 
        # are assembly biased. 
        self._create_haloprop_keys()

        # In MCMC applications, the output_dict items define the 
        # parameter set explored by the likelihood engine. 
        # Changing the values of the parameters in param_dict 
        # will propagate to the behavior of the component models, 
        # though the param_dict attributes attached to the component model 
        # instances themselves will not be changed. 
        self.param_dict = (
            self.retrieve_all_inherited_parameters(
                self.component_model_dict)
            )

        self.publications = self.retrieve_all_publications(
            self.component_model_dict)

    def mean_occupation(self,gal_type,*args):
        """ Method supplying the mean abundance of gal_type galaxies. 
        The behavior of this method is inherited from one of the component models. 
        """

        self.test_component_consistency(gal_type,'occupation_model')

        # For galaxies of type gal_type, the behavior of this method 
        # will be set by the inherited occupation_model object 
        occupation_model = self.component_model_dict[gal_type]['occupation_model']
        inherited_method = occupation_model.mean_occupation

        output_occupation = self.retrieve_component_behavior(inherited_method,args)

        return output_occupation


    def mc_occupation(self,gal_type,halo_table):
        """ Method providing a Monte Carlo realization of the mean occupation.
        The behavior of this method is inherited from one of the component models.
        """

        # Retrieve the appropriate columns from halo_table
        input_haloprops = [halo_table[self.prim_haloprop_key]]
        assembias_occupation = ((gal_type in self.sec_haloprop_key_dict.keys()) & 
            ('occupation_model' in self.sec_haloprop_key_dict[gal_type]) )
        if assembias_occupation == True:
            input_haloprops.extend([halo_table[self.sec_haloprop_key]])

        self.test_component_consistency(gal_type,'occupation_model')

        occupation_model = self.component_model_dict[gal_type]['occupation_model']
        inherited_method = occupation_model.mc_occupation
        output_mc_realization = self.retrieve_component_behavior(inherited_method,input_haloprops)

        return output_mc_realization
        
    def mean_profile_parameters(self,gal_type,*args):
        """ Method returning the mean value of the parameters governing the radial profile 
        of gal_type galaxies. 
        The behavior of this method is inherited from one of the component models.
        """

        self.test_component_consistency(gal_type,'profile_model')

        profile_model = self.component_model_dict[gal_type]['profile_model']
        inherited_method = profile_model.mean_profile_parameters
        output_profiles = self.retrieve_component_behavior(inherited_method,args)

        return output_profiles

    def mc_coords(self,gal_type,*args):
        """ Method returning a Monte Carlo realization of the radial profile. 
        The behavior of this method is inherited from one of the component models.
        """

        self.test_component_consistency(gal_type,'profile_model')

        profile_model = self.component_model_dict[gal_type]['profile_model']
        inherited_method = profile_model.mc_coords
        output_mc_realization = self.retrieve_component_behavior(inherited_method,args)

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

    def retrieve_component_behavior(self,inherited_method,*args):
        """ Wrapper method whose purpose is solely to call the component model methods 
        using the correct number of arguments. Purely for user convenience. 

        """

        if len(args)==1:
            prim_haloprop = args[0]
            output = inherited_method(prim_haloprop)
        elif len(args)==2:
            prim_haloprop, sec_haloprop = args[0], args[1]
            output = inherited_method(prim_haloprop,sec_haloprop)
        else:
            raise TypeError("Only one or two halo property inputs are supported by "
                "mean_occupation method")

        return output

    def retrieve_all_inherited_parameters(self,component_model_dict, 
        testmode=defaults.testmode):
        """ Method to build a dictionary of parameters for the composite model 
        by retrieving all the parameters of the component models. 

        Parameters 
        ----------
        component_model_dict : dict 
            Dictionary passed to the HOD factory __init__ constructor 
            that is used to provide instructions for how to build a 
            composite model from a set of components. 

        Returns 
        -------
        output_dict : dict 
            Dictionary of all parameters used by all component models. 
        """

        output_dict = {}

        # Loop over all galaxy types in the composite model
        for gal_type_dict in component_model_dict.values():
            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():

                if testmode == True:
                    occuhelp.test_repeated_keys(
                        output_dict, model_instance.param_dict)

                output_dict = dict(
                    model_instance.param_dict.items() + 
                    output_dict.items())

        return output_dict

    def retrieve_all_publications(self, component_model_dict):
        """ Method to build a list of publications 
        associated with each component model. 

        Parameters 
        ----------
        component_model_dict : dict 
            Dictionary passed to the HOD factory __init__ constructor 
            that is used to provide instructions for how to build a 
            composite model from a set of components. 

        Returns 
        -------
        pub_list : array_like 
        """
        pub_list = []

        # Loop over all galaxy types in the composite model
        for gal_type_dict in component_model_dict.values():
            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():
                pub_list.extend(model_instance.publications)

        return pub_list

    def retrieve_relevant_haloprops(self, *args, **kwargs):
        """ Method returning the arrays that need to be passed 
        to the component model in order to access its behavior. 
        """

        if ( (occuhelp.aph_len(args) == 0) & ('mock_galaxies' in kwargs.keys()) ):
            pass
        elif ( (occuhelp.aph_len(args) > 0) & ('mock_galaxies' not in kwargs.keys()) ):
            return args
        elif ( (occuhelp.aph_len(args) == 0) & ('mock_galaxies' not in kwargs.keys()) ):
            raise SyntaxError("Neither an array of halo properties "
                " nor a mock galaxy population was passed")
        else ( (occuhelp.aph_len(args) > 0) & ('mock_galaxies' in kwargs.keys()) ):
            raise SyntaxError("Do not pass both an array of halo properties "
                " and a mock galaxy population - pick one")







    def _create_haloprop_keys(self):

        # Create attribute for primary halo property used by all component models
        # Forced to be the same property defining the underlying halo profile 
        # seen by all galaxy types 
        self.prim_haloprop_key = self.halo_prof_model.prim_haloprop_key

        # If any of the galaxy types have any assembly-biased component behavior, 
        # create a second attribute called sec_haloprop_key. 
        # Force the secondary halo property to be the same for all behaviors 
        # of all galaxy types. May wish to relax this requirement later. 
        sec_haloprop_key_dict = {}
        for gal_type in self.gal_types:
            temp_dict = {}
            for behavior_key, behavior_model in self.component_model_dict[gal_type].iteritems():
                if hasattr(behavior_model,'sec_haloprop_key'):
                    temp_dict[behavior_key] = behavior_model.sec_haloprop_key
            if len(set(temp_dict.values())) > 1:
                raise KeyError("If implementing assembly bias for a particular gal_type, "
                    "must use the same secondary halo property "
                    " for all behaviors of this galaxy type")
            elif len(set(temp_dict.values())) == 1:
                sec_haloprop_key_dict[gal_type] = temp_dict
        if len(set(sec_haloprop_key_dict.values())) > 1:
            raise KeyError("If implementing assembly bias in a composite model, "
                " must use same secondary halo property for all galaxy types")
        elif len(set(sec_haloprop_key_dict.values())) == 1:
            self.sec_haloprop_key = sec_haloprop_key_dict.values()[0]
            self.sec_haloprop_key_dict = sec_haloprop_key_dict


















