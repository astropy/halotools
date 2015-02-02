# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
composite HOD models from a set of components. 

"""

__all__ = ['HodModel']

import numpy as np
import occupation_helpers as occuhelp
import defaults

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

        # Need to have a _example_attr_dict attribute. See end of 
        # mock_factory._allocate_memory()

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
        # Also create a dictionary for which gal_types, and which behaviors, 
        # are assembly-biased. 
        self._create_haloprop_keys()

        # In MCMC applications, the output_dict items define the 
        # parameter set explored by the likelihood engine. 
        # Changing the values of the parameters in param_dict 
        # will propagate to the behavior of the component models, 
        # though the param_dict attributes attached to the component model 
        # instances themselves will not be changed. 
        self.param_dict = (
            self.build_composite_param_dict(
                self.component_model_dict)
            )

        self.publications = self.build_publication_list(
            self.component_model_dict)

    def retrieve_relevant_haloprops(self, gal_type, component_key, 
        *args, **kwargs):
        """ Method returning the arrays that need to be passed 
        to a component model in order to access its behavior. 
        """

        if ( (occuhelp.aph_len(args) == 0) & ('mock_galaxies' in kwargs.keys()) ):
            # In this case, we were passed a full mock galaxy catalog as a keyword argument
            mock = kwargs['mock_galaxies']
            # each component model has a dictionary containing the keys of the 
            # halo catalog that the component model needs from the halo catalog
            haloprop_key_dict = self.component_model_dict[gal_type][component_key].haloprop_key_dict
            # All such dictionaries have a key indicating the primary halo property governing the behavior
            prim_haloprop_key = haloprop_key_dict['prim_haloprop_key']
            # We were passed the full mock, but this function call only pertains to the slice of 
            # the arrays that correspond to gal_type galaxies. 
            # We save time by having pre-computed the relevant slice. 
            gal_type_slice = mock._gal_type_indices[gal_type]
            prim_haloprop = getattr(mock, prim_haloprop_key)[gal_type_slice]
            # Now pack the prim_haloprop array into a 1-element list
            output_columns = [prim_haloprop]
            # If there is a secondary halo property used by this component model, 
            # repeat the above retrieval and extend the list. 
            if 'sec_haloprop_key' in haloprop_key_dict.keys():
                sec_haloprop_key = haloprop_key_dict['sec_haloprop_key']
                sec_haloprop = getattr(mock, sec_haloprop_key)[gal_type_slice]
                output_columns.extend([sec_haloprop])

            return output_columns

        elif ( (occuhelp.aph_len(args) > 0) & ('mock_galaxies' not in kwargs.keys()) ):
            # In this case, we were directly passed the relevant arrays
            return list(args)
        ###
        ### Now address the cases where we were passed insensible arguments
        elif ( (occuhelp.aph_len(args) == 0) & ('mock_galaxies' not in kwargs.keys()) ):
            raise SyntaxError("Neither an array of halo properties "
                " nor a mock galaxy population was passed")
        else:
            raise SyntaxError("Do not pass both an array of halo properties "
                " and a mock galaxy population - pick one")

    def mean_occupation(self,gal_type,*args, **kwargs):
        """ Method supplying the mean abundance of gal_type galaxies. 
        The behavior of this method is inherited from one of the component models. 
        """

        component_key = 'occupation_model'
        method_name = 'mean_occupation'

        output_occupation = self.retrieve_component_behavior(
            gal_type, component_key, method_name, *args, **kwargs)

        return output_occupation


    def mc_occupation(self,gal_type,*args, **kwargs):
        """ Method providing a Monte Carlo realization of the mean occupation.
        The behavior of this method is inherited from one of the component models.
        """

        component_key = 'occupation_model'
        method_name = 'mc_occupation'

        output_mc_realization = self.retrieve_component_behavior(
            gal_type, component_key, method_name, *args, **kwargs)

        return output_mc_realization
        
    def mean_profile_parameters(self,gal_type,*args, **kwargs):
        """ Method returning the mean value of the parameters governing the radial profile 
        of gal_type galaxies. 
        The behavior of this method is inherited from one of the component models.
        """

        profile_model = self.component_model_dict[gal_type]['profile_model']
        inherited_method = profile_model.mean_profile_parameters
        output_profiles = self.retrieve_component_behavior(inherited_method,args)

        return output_profiles

    def mc_coords(self,gal_type,*args, **kwargs):
        """ Method returning a Monte Carlo realization of the radial profile. 
        The behavior of this method is inherited from one of the component models.
        """

        profile_model = self.component_model_dict[gal_type]['profile_model']
        inherited_method = profile_model.mc_coords
        output_mc_realization = self.retrieve_component_behavior(inherited_method,args)

        return output_mc_realization


    def retrieve_component_behavior(self, gal_type, component_key, method_name, 
        *args, **kwargs):
        """ Wrapper method whose purpose is solely to call the component model methods 
        using the correct number of arguments. Purely for user convenience. 

        """

        # The behavior of mc_occupation is inherited by the component model 
        component_model_instance = self.component_model_dict[gal_type][component_key]
        inherited_method = getattr(component_model_instance, method_name)

        # Retrieve the appropriate columns from halo_table
        haloprop_list = self.retrieve_relevant_haloprops(
            gal_type, component_key, *args, **kwargs)
        # haloprop_list is a one- or two-element list of arrays of halo properties. 
        # Use the * syntax to unpack this list into a sequence of positional arguments. 
        output = inherited_method(*haloprop_list)

        return output
 

    def build_composite_param_dict(self,component_model_dict):
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

                occuhelp.test_repeated_keys(
                    output_dict, model_instance.param_dict)

                output_dict = dict(
                    model_instance.param_dict.items() + 
                    output_dict.items()
                    )

        return output_dict

    def build_publication_list(self, component_model_dict):
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


















