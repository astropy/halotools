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
    model_blueprint, which serves as a set of instructions for how the 
    composite model is to be built from the components. 

    """

    def __init__(self, halo_prof_model, model_blueprint, 
        haloprop_key_dict=defaults.haloprop_key_dict):
        """ The methods of this class derive their behavior from other, external classes, 
        passed in the form of the model_blueprint, a dictionary whose keys 
        are the galaxy types found in the halos, e.g., 'centrals', 'satellites', 'orphans', etc.
        The values of the model_blueprint are themselves dictionaries whose keys  
        specify the type of model being passed, e.g., 'occupation_model', and values 
        are instances of that type of model. The model_blueprint dictionary is built by 
        the hod_designer interface. The input halo_prof_model is an instance of the class 
        governing the assumed profile of the underlying halos. 

        """

        # Bind the model-building instructions to the composite model
        self.halo_prof_model = halo_prof_model
        self.model_blueprint = model_blueprint

        self.haloprop_key_dict = haloprop_key_dict
        self.prim_haloprop_key = self.haloprop_key_dict['prim_haloprop_key']
        if 'sec_haloprop_key' in self.haloprop_key_dict.keys():
            self.sec_haloprop_key = self.haloprop_key_dict['sec_haloprop_key']

        # Create attributes for galaxy types and their occupation bounds
        self.set_gal_types()

        # Build the composite model parameter dictionary by retrieving 
        # the relevant items from the component model dictionaries. 
        self.param_dict = self.build_composite_param_dict(self.model_blueprint)

        self.publications = self.build_publication_list(
            self.model_blueprint)

    def set_gal_types(self):
        """ Method creates attributes for the type of galaxies in the model 
        and the upper bound on their per-halo abundance. 
        """

        self.gal_types = self.model_blueprint.keys()
        self.occupation_bound = {}
        for gal_type in self.gal_types:
            self.occupation_bound[gal_type] = (
                self.model_blueprint[gal_type]['occupation_model'].occupation_bound)

    def gal_prof_param(self, gal_type, gal_prof_param_key, mock_galaxies):
        """ If the galaxy profile model has gal_prof_param_key as a biased parameter, 
        call the galaxy profile model. Otherwise, return the value of the halo profile parameter. 
        """

        method_name = gal_prof_param_key+'_'+gal_type
        if hasattr(self, method_name):
            pass
        else:
            halo_prof_param_key = (
                defaults.host_haloprop_prefix + 
                gal_prof_param_key[len(defaults.galprop_prefix):]
                )
            gal_type_slice = mock_galaxies._gal_type_indices[gal_type]
            return getattr(mock_galaxies, halo_prof_param_key)[gal_type_slice]

    def set_primary_behaviors(self):

        for gal_type in self.gal_types:

            # First set a method for each profile parameter
            gal_prof_model = self.model_blueprint[gal_type]['profile_model']


            for gal_prof_param in self.gal_prof_model.prof_param_keys:
                behavior_name = gal_prof_param+'_'+gal_type
                component_instance = self.model_blueprint[gal_type]['profile_model']
                component_behavior = getattr(component_instance, behavior_name)
                setattr(self, behavior_name, component_behavior)

            behavior_name = 'pos_'+gal_type
            component_instance = self.model_blueprint[gal_type]['pos']
            component_behavior = getattr(component_instance, 'mc_pos')
            setattr(self, behavior_name, component_behavior)

    def set_halo_prof_func_dict(self):
        """ Method to derive the halo profile parameter function dictionary 
        from a collection of galaxies. 

        Notes 
        -----
        If there are multiple instances of the same underlying 
        halo profile model, such as would happen if 
        there are two satellite-like populations with NFW profiles, 
        only one will be used and a warning will be issued. 
        """

        self.halo_prof_func_dict = {}
        tmp_key_correspondence = {}
        for gal_type in self.gal_types:
            gal_prof_model = self.model_blueprint[gal_type]['profile_model']
            halo_prof_func_dict = gal_prof_model.halo_prof_model.halo_prof_func_dict
            for key in halo_prof_func_dict.keys():
                if key not in self.halo_prof_func_dict.keys():
                    # Set the profile function for this parameter
                    self.halo_prof_func_dict[key] = halo_prof_func_dict[key]
                    # Bookkeeping device to manage potential key repetition
                    tmp_key_correspondence[key] = gal_type
                else:
                    msg = "The halo profile parameter function %s\n"
                    "appears in the halo profile model associated with both\n"
                    "%s and %s. \nIgnoring the %s model and using the %s model\n"
                    "to compute the new halo catalog column %s"
                    ignored_gal_type = gal_type
                    relevant_gal_type = tmp_key_correspondence[key]
                    print(msg % (key, ignored_gal_type, relevant_gal_type, 
                        ignored_gal_type, relevant_gal_type, key))

    def set_prof_param_table_dict(self,input_dict=None):

        # Set all profile parameter table dictionaries. 
        # Also set which builder function will be used by the composite 
        # method build_inv_cumu_lookup_table

        # At the end, if passed an input_dict,
        # we will use it to over-write with the keys present in input_dict

        self.prof_param_table_dict = {}
        #self.prof_param_table_builder_dict = {}
        self._gal_type_prof_param_key_correspondence = {}

        for gal_type in self.gal_types:
            gal_prof_model = self.model_blueprint[gal_type]['profile_model']
            prof_param_table_dict = gal_prof_model.halo_prof_model.prof_param_table_dict
            for key in prof_param_table_dict.keys():
                if key not in self.prof_param_table_dict.keys():
                    # Set the gridding scheme for this parameter
                    self.prof_param_table_dict[key] = prof_param_table_dict[key]
                    # Set the table builder function for this parameter
                    #self.prof_param_table_builder_dict[key] = (
                    #    gal_prof_model.halo_prof_model.build_inv_cumu_lookup_table
                    #    )
                    # Bookkeeping device to manage potential key repetition
                    self._gal_type_prof_param_key_correspondence[key] = gal_type
                else:
                    msg = "The halo profile parameter %s\n"
                    "appears in the halo profile model associated with both\n"
                    "%s and %s. \nIgnoring the %s model and using the %s model\n"
                    "to build prof_param_table_dict %s"
                    ignored_gal_type = gal_type
                    relevant_gal_type = self._gal_type_prof_param_key_correspondence[key]
                    print(msg % (key, ignored_gal_type, relevant_gal_type, 
                        ignored_gal_type, relevant_gal_type, key))
 
        # Finally, use input_dict to overwrite table values, if applicable
        if input_dict != None:
            for key, table in input_dict.iteritems():
                if type(table) is not tuple:
                    raise TypeError("input_dict must have tuples for values")
                if len(table) is not 3:
                    raise TypeError("Length of input_dict tuple must be 3")
                self.prof_param_table_dict[key] = table


    def build_inv_cumu_lookup_table(self, prof_param_table_dict={}):

        self.set_prof_param_table_dict(prof_param_table_dict)

        self.cumu_inv_func_table_dict = {}
        self.cumu_inv_param_table_dict = {}

        for key in self.prof_param_table_dict.keys():
            gal_type = self._gal_type_prof_param_key_correspondence[key]
            gal_prof_model = self.model_blueprint[gal_type]['profile_model']
            builder = gal_prof_model.halo_prof_model.build_inv_cumu_lookup_table

            if key in prof_param_table_dict.keys():
                builder(prof_param_table_dict)
            else:
                builder()

            self.cumu_inv_func_table_dict[key] = (
                gal_prof_model.halo_prof_model.cumu_inv_func_table)
            self.cumu_inv_param_table_dict[key] = (
                gal_prof_model.halo_prof_model.cumu_inv_param_table)


    def component_behavior(self, gal_type, colname, *args, **kwargs):

        relevant_data = self.retrieve_relevant_haloprops(
            gal_type, colname, *args, **kwargs)

        component_model_function = getattr(self, colname='_'+gal_type)

        return component_model_function(*relevant_data, 
            input_param_dict=self.param_dict)


    def retrieve_relevant_haloprops(self, gal_type, component_key, 
        *args, **kwargs):
        """ Method returning the arrays that need to be passed 
        to a component model in order to access its behavior. 

        Parameters 
        ----------
        gal_type : string 

        component_key : string 
            Key used to access the behavior of the component model. 
            component_key must be an element prim_func_dict.keys(), 
            a dictionary bound to every component model. 

        prim_haloprop : array_like, optional positional argument

        sec_haloprop : array_like, optional positional argument

        mock_galaxies : object, optional keyword argument 

        Returns 
        -------
        result : list 
            List of arrays of the relevant halo properties

        """

        if ( (occuhelp.aph_len(args) == 0) & ('mock_galaxies' in kwargs.keys()) ):
            # In this case, we were passed a full mock galaxy catalog as a keyword argument
            mock = kwargs['mock_galaxies']

            prim_haloprop_key = self.haloprop_key_dict['prim_haloprop_key']
            # We were passed the full mock, but this function call only pertains to the slice of 
            # the arrays that correspond to gal_type galaxies. 
            # We save time by having pre-computed the relevant slice. 
            gal_type_slice = mock._gal_type_indices[gal_type]
            prim_haloprop = getattr(mock, prim_haloprop_key)[gal_type_slice]
            # Now pack the prim_haloprop array into a 1-element list
            output_columns = [prim_haloprop]
            # If there is a secondary halo property used by this component model, 
            # repeat the above retrieval and extend the list. 
            if 'sec_haloprop_key' in self.haloprop_key_dict.keys():
                sec_haloprop_key = self.haloprop_key_dict['sec_haloprop_key']
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


    def build_composite_param_dict(self,model_blueprint):
        """ Method to build a dictionary of parameters for the composite model 
        by retrieving all the parameters of the component models. 

        In MCMC applications, the output_dict items define the 
        parameter set explored by the likelihood engine. 
        Changing the values of the parameters in param_dict 
        will propagate to the behavior of the component models. 
        Note, though, that the param_dict attributes attached to the component model 
        instances themselves will not be changed. 

        Parameters 
        ----------
        model_blueprint : dict 
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
        for gal_type_dict in model_blueprint.values():
            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():

                occuhelp.test_repeated_keys(
                    output_dict, model_instance.param_dict)

                output_dict = dict(
                    model_instance.param_dict.items() + 
                    output_dict.items()
                    )

        return output_dict

    def build_publication_list(self, model_blueprint):
        """ Method to build a list of publications 
        associated with each component model. 

        Parameters 
        ----------
        model_blueprint : dict 
            Dictionary passed to the HOD factory __init__ constructor 
            that is used to provide instructions for how to build a 
            composite model from a set of components. 

        Returns 
        -------
        pub_list : array_like 
        """
        pub_list = []

        # Loop over all galaxy types in the composite model
        for gal_type_dict in model_blueprint.values():
            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():
                pub_list.extend(model_instance.publications)

        return list(set(pub_list))

    def _create_convenience_attributes(self):
        """ Create attributes of the composite model to conveniently access 
        the most commonly used methods of the component models. 
        """

        for gal_type, gal_type_dict in self.model_blueprint.iteritems():
            for component_key, component_instance in gal_type_dict.iteritems():
                # First create a convenience method for each entry in 
                # the primary function dictionary
                for method in component_instance.prim_func_dict.values():
                    method_name = method.__name__+'_'+component_instance.gal_type
                    setattr(self, method_name, method)
                # If the component has additional methods 
                # we'd like convenience attributes for, create those too.
                if hasattr(component_instance, 'additional_methods_to_inherit'):
                    convenience_methods = component_instance.additional_methods_to_inherit
                for method in convenience_methods:
                    method_name = method.__name__+'_'+component_instance.gal_type
                    setattr(self, method_name, method)






























