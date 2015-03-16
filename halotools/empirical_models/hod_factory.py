# -*- coding: utf-8 -*-
"""
Module containing the primary class used to build 
composite HOD-style models from a set of components. 
"""

__all__ = ['HodModelFactory']

from functools import partial
import numpy as np
import occupation_helpers as occuhelp
import model_defaults

class HodModelFactory(object):
    """ This is the composite HOD model object that can be used to generate 
    mock galaxy populations. The primary methods are for assigning 
    the mean occupation of a galaxy population 
    to a halo, the intra-halo radial profile of that population, and the 
    accompanying methods to generate Monte Carlo realizations of those methods. 

    All behavior is derived from external classes passed to the constructor, 
    so this HOD Model Factory does nothing more than compose these external 
    behaviors into a standardized form that talks to the rest of the package. 
    The way this works is the Model Factory gets passed a dictionary that serves as 
    a blueprint for how to build the model. The building of that dictionary is done 
    elsewhere, by the `~halotools.empirical_models.hod_designer` class. So this class 
    receives these blueprint dictionaries as input, and returns a model object 
    that can directly populate simulations with mock populations. 

    """

    def __init__(self, model_blueprint):
        """ The methods of this class derive their behavior from other, external classes, 
        passed in the form of the model_blueprint, a dictionary whose keys 
        are the galaxy types found in the halos, e.g., 'centrals', 'satellites', 'orphans', etc.
        The values of the model_blueprint are themselves dictionaries whose keys  
        specify the type of model being passed, e.g., 'occupation', and values 
        are class instances of that type of model. The input halo_prof_model is an instance of the class 
        governing the assumed profile of the underlying halos. 

        """

        # Bind the model-building instructions to the composite model
        self.model_blueprint = model_blueprint

        # Determine the halo properties governing the galaxy population properties
        self._set_haloprops()

        # Create attributes for galaxy types and their occupation bounds
        self._set_gal_types()

        # Build the composite model dictionary, whose keys are parameters of our model
        self._set_init_param_dict()

        # Determine the functions that will be used
        # to map halo profile parameters onto halos
        self._set_halo_prof_func_dict()
        self._set_prof_param_table_dict()

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self._set_primary_behaviors()

        self.publications = self._build_publication_list()

    def _set_haloprops(self):

        self.haloprop_key_dict = return_haloprop_dict(self.model_blueprint)

        occuhelp.enforce_required_haloprops(self.haloprop_key_dict)
        self.prim_haloprop_key = self.haloprop_key_dict['prim_haloprop_key']
        if 'sec_haloprop_key' in self.haloprop_key_dict.keys():
            self.sec_haloprop_key = self.haloprop_key_dict['sec_haloprop_key']

    def _set_gal_types(self):
        gal_types = self.model_blueprint.keys()

        occupation_bounds = []
        for gal_type in gal_types:
            model = self.model_blueprint[gal_type]['occupation']
            occupation_bounds.append(model.occupation_bound)

        sorted_idx = np.argsort(occupation_bounds)
        gal_types = list(np.array(gal_types)[sorted_idx])
        self.gal_types = gal_types

        self.occupation_bound = {}
        for gal_type in self.gal_types:
            self.occupation_bound[gal_type] = (
                self.model_blueprint[gal_type]['occupation'].occupation_bound)

    def gal_prof_param(self, gal_type, gal_prof_param_key, mock_galaxies):
        """ If the galaxy profile model has gal_prof_param_key as a biased parameter, 
        call the galaxy profile model. Otherwise, return the value of the halo profile parameter. 
        """

        gal_type_slice = mock_galaxies._gal_type_indices[gal_type]

        method_name = gal_prof_param_key+'_'+gal_type            

        if hasattr(self, method_name):
            # column name of mock_galaxies containing the profile parameter values 
            # We are calling the component SpatialBias model, 
            # so there should be no 'halo_' prefix
            halo_prof_param_key = gal_prof_param_key[len(model_defaults.galprop_prefix):]
            result = self.method_name(halo_prof_param_key, 
                getattr(mock_galaxies, self.prim_haloprop_key)[gal_type_slice],
                getattr(mock_galaxies, halo_prof_param_key)[gal_type_slice]
                )
            return result
        else:
            # column name of mock_galaxies containing the profile parameter values 
            # We are accessing the existing column of mock_galaxies,
            # so in this case there should be a 'halo_' prefix 
            halo_prof_param_key = (
                model_defaults.host_haloprop_prefix + 
                gal_prof_param_key[len(model_defaults.galprop_prefix):]
                )
            return getattr(mock_galaxies, halo_prof_param_key)[gal_type_slice]

    def _set_primary_behaviors(self):
        """ This function creates a bunch of new methods that it binds to ``self``. 
        These methods are given standardized names, for generic communication with 
        the rest of the package, particularly the *Mock Factory*. 
        The behaviors of these methods are defined elsewhere; 
        here we just create a link to those external behaviors. 
        """

        for gal_type in self.gal_types:

            # Set the method used to Monte Carlo realizations of per-halo gal_type abundance
            new_method_name = 'mc_occupation_'+gal_type
            occupation_model = self.model_blueprint[gal_type]['occupation']
            new_method_behavior = partial(occupation_model.mc_occupation, 
                input_param_dict = self.param_dict)
            setattr(self, new_method_name, new_method_behavior)

            # Also inherit the mean abundance, for convenience
            new_method_name = 'mean_occupation_'+gal_type
            new_method_behavior = partial(occupation_model.mean_occupation, 
                input_param_dict = self.param_dict)
            setattr(self, new_method_name, new_method_behavior)

            # Now move on to galaxy profiles
            gal_prof_model = self.model_blueprint[gal_type]['profile']

            # We will loop over gal_prof_model.gal_prof_func_dict
            # This dictionary only contains keys for biased profile parameters
            # Thus there will be no new methods created for unbiased profile parameters
            for gal_prof_param_key, gal_prof_param_func in (
                gal_prof_model.gal_prof_func_dict.iteritems()):
                new_method_name = gal_prof_param_key+'_'+gal_type
                new_method_behavior = gal_prof_param_func
                setattr(self, new_method_name, new_method_behavior)

            new_method_name = 'pos_'+gal_type
            new_method_behavior = partial(self.mc_pos, gal_type = gal_type)
            setattr(self, new_method_name, new_method_behavior)

    def mc_pos(self, mock_galaxies, gal_type):
        """ Method used to generate Monte Carlo realizations of galaxy positions. 

        Identical to component model version from which the behavior derives, 
        only this method re-scales the halo-centric distance by the halo radius, 
        and re-centers the re-scaled output of the component model to the halo position.

        """
        gal_prof_model = self.model_blueprint[gal_type]['profile']
        mc_pos_function = getattr(gal_prof_model, 'mc_pos')

        output_pos = mc_pos_function(mock_galaxies)

        gal_type_slice = mock_galaxies._gal_type_indices[gal_type]

        # Re-scale the halo-centric distance by the halo boundary
        halo_boundary_attr_name = (
            model_defaults.host_haloprop_prefix + 
            model_defaults.haloprop_key_dict['halo_boundary']
            )

        for idim in range(3): 
            output_pos[:,idim] *= getattr(mock_galaxies, halo_boundary_attr_name)[gal_type_slice]

        # Re-center the positions by the host halo location
        halo_pos_attr_name = model_defaults.host_haloprop_prefix+'pos'
        output_pos += getattr(mock_galaxies, halo_pos_attr_name)[gal_type_slice]

        return output_pos


    def _set_halo_prof_func_dict(self):
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
            gal_prof_model = self.model_blueprint[gal_type]['profile']
            tmp_halo_prof_func_dict = gal_prof_model.halo_prof_model.halo_prof_func_dict
            for key in tmp_halo_prof_func_dict.keys():
                if key not in self.halo_prof_func_dict.keys():
                    # Set the profile function for this parameter
                    self.halo_prof_func_dict[key] = tmp_halo_prof_func_dict[key]
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

        # Finally, create a convenience list of galaxy profile parameter keys
        # This list is identical to self.halo_prof_func_dict.keys(), 
        # but pre-pended by model_defaults.galprop_prefix
        self._set_gal_prof_params()

    def _set_gal_prof_params(self):
        self.gal_prof_param_keys = []
        for key in self.halo_prof_func_dict.keys():
            galkey = model_defaults.galprop_prefix+key
            self.gal_prof_param_keys.append(galkey)


    def _set_prof_param_table_dict(self,input_dict=None):

        # Set all profile parameter table dictionaries. 
        # Also set which builder function will be used by the composite 
        # method build_inv_cumu_lookup_table

        # At the end, if passed an input_dict,
        # we will use it to over-write with the keys present in input_dict

        self.prof_param_table_dict = {}
        #self.prof_param_table_builder_dict = {}
        self._gal_type_prof_param_key_correspondence = {}

        for gal_type in self.gal_types:
            gal_prof_model = self.model_blueprint[gal_type]['profile']
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

        self._set_prof_param_table_dict(prof_param_table_dict)

        self.cumu_inv_func_table_dict = {}
        self.cumu_inv_param_table_dict = {}

        for key in self.prof_param_table_dict.keys():
            gal_type = self._gal_type_prof_param_key_correspondence[key]
            gal_prof_model = self.model_blueprint[gal_type]['profile']
            builder = gal_prof_model.halo_prof_model.build_inv_cumu_lookup_table

            if key in prof_param_table_dict.keys():
                builder(prof_param_table_dict)
            else:
                builder()

            self.cumu_inv_func_table_dict[key] = (
                gal_prof_model.halo_prof_model.cumu_inv_func_table)
            self.cumu_inv_param_table_dict[key] = (
                gal_prof_model.halo_prof_model.cumu_inv_param_table)

    def retrieve_relevant_haloprops(self, gal_type, *args, **kwargs):
        """ Method returning the arrays that need to be passed 
        to a component model in order to access its behavior. 

        Parameters 
        ----------
        gal_type : string 

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


    def _set_init_param_dict(self):
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

        """

        self.param_dict = {}

        # Loop over all galaxy types in the composite model
        for gal_type_dict in self.model_blueprint.values():
            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():

                occuhelp.test_repeated_keys(
                    self.param_dict, model_instance.param_dict)

                self.param_dict = dict(
                    model_instance.param_dict.items() + 
                    self.param_dict.items()
                    )

    def update_param_dict(self, new_param_dict):
        pass

    def _build_publication_list(self):
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
        for gal_type_dict in self.model_blueprint.values():

            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():
                pub_list.extend(model_instance.publications)

        return list(set(pub_list))



##########################################

def return_haloprop_dict(model_blueprint):

    prim_haloprop_list = []
    sec_haloprop_list = []
    halo_boundary_list = []
    
    no_prim_haloprop_msg = "For gal_type %s and feature %s, no primary haloprop detected"
    no_halo_boundary_msg = "For gal_type %s, no primary haloprop detected for profile model"

    for gal_type in model_blueprint.keys():
        for feature in model_blueprint[gal_type].values():

            if 'prim_haloprop_key' in feature.haloprop_key_dict.keys():
                prim_haloprop_list.append(feature.haloprop_key_dict['prim_haloprop_key'])
            else:
                print(no_prim_haloprop_msg % (gal_type, feature))

            if 'sec_haloprop_key' in feature.haloprop_key_dict.keys():
                sec_haloprop_list.append(feature.haloprop_key_dict['sec_haloprop_key'])

            if 'halo_boundary' in feature.haloprop_key_dict.keys():
                halo_boundary_list.append(feature.haloprop_key_dict['halo_boundary'])

    if len(set(prim_haloprop_list)) == 0:
        raise KeyError("No component feature of any gal_type had a prim_haloprop")
    elif len(set(prim_haloprop_list)) > 1:
        raise KeyError("Distinct prim_haloprop choices for different feature"
            " is not supported")

    if len(set(halo_boundary_list)) == 0:
        raise KeyError("No component feature of any gal_type had a halo_boundary")
    elif len(set(halo_boundary_list)) > 1:
        raise KeyError("Distinct halo_boundary choices for different gal_types"
            " is not supported")

    if len(set(sec_haloprop_list)) > 1:
        raise KeyError("Distinct prim_haloprop choices for different feature"
            " is not supported")

    output_dict = {}
    output_dict['prim_haloprop_key'] = prim_haloprop_list[0]
    output_dict['halo_boundary'] = halo_boundary_list[0]
    if sec_haloprop_list != []:
        output_dict['sec_haloprop_key'] = sec_haloprop_list[0]

    return output_dict





""" # Not sure whether this is worth doing

    def _create_convenience_attributes(self):

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

"""


























