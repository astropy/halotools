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
    """ Class used to build HOD-style models of the galaxy-halo connection. 

    Can be thought of as a factory that takes an HOD model blueprint as input, 
    and generates an HOD Model object. The returned object can be used directly to 
    populate a simulation with a Monte Carlo realization of the model. 

    All behavior is derived from external classes bound up in the input blueprint. 
    So `HodModelFactory` does nothing more than compose these external 
    behaviors together into a composite model. 
    The primary purpose is to provide a standardized model object 
    that interfaces consistently with the rest of the package, 
    regardless of the features of the model. 

    The building of the blueprint is done elsewhere. Pre-loaded blueprints 
    can be found in `~halotools.empirical_models.preloaded_hod_blueprints`, 
    or you can also design your own from scratch. 

    Parameters 
    ----------
    model_blueprint : dict 
        Dictionary keys of ``model_blueprint`` are the names of the types of galaxies 
        found in the halos, e.g., ``centrals``, ``satellites``, ``orphans``, etc. 
        Dictionary values of ``model_blueprint`` are themselves dictionaries whose keys 
        specify the type of model being passed, e.g., ``occupation``, 
        and values are class instances of that type of model.

    """

    def __init__(self, model_blueprint):

        # Bind the model-building instructions to the composite model
        self.model_blueprint = model_blueprint

        # Determine the halo properties governing the galaxy population properties
        occuhelp.enforce_required_haloprops(self.haloprop_key_dict)

        self.prim_haloprop_key = self.haloprop_key_dict['prim_haloprop_key']
        if 'sec_haloprop_key' in self.haloprop_key_dict.keys():
            self.sec_haloprop_key = self.haloprop_key_dict['sec_haloprop_key']

        # Create attributes for galaxy types and their occupation bounds
        self._set_gal_types()

        # Build the composite model dictionary, whose keys are parameters of our model
        self._set_init_param_dict()

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self._set_primary_behaviors()

        self.publications = self._build_publication_list()


    @property 
    def haloprop_key_dict(self):
        """ Dictionary defining the halo properties 
        that regulate galaxy occupation statistics. 

        Dict keys always include ``prim_haloprop_key`` and ``halo_boundary``, 
        whose default settings are defined in `~halotools.empirical_models.model_defaults`. 
        Models with assembly bias will include a ``sec_haloprop_key`` key. 
        Dict values are strings used to access the appropriate column of a halo catalog, 
        e.g., ``mvir``. 
        """

        return return_haloprop_dict(self.model_blueprint)

    def _set_gal_types(self):
        """ Private method binding the ``gal_types`` list attribute,
        and the ``occupation_bound`` attribute, to the class instance. 
        List is sequenced in ascending order of the occupation bound. 
        """

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


    @property 
    def gal_prof_param_list(self):
        """ List of all galaxy profile parameters used by the composite model. 
        """

        output_list = []
        for gal_type in self.gal_types:
            gal_prof_model = self.model_blueprint[gal_type]['profile']
            output_list.extend(gal_prof_model.gal_prof_func_dict.keys())
        output_list = list(set(output_list))

        return output_list


    def _set_primary_behaviors(self):
        """ This function creates a bunch of new methods that it binds to ``self``. 
        These methods are given standardized names, for generic communication with 
        the rest of the package, particularly the `HodMockFactory`. 
        The behaviors of these methods are defined elsewhere; 
        here we just create a link to those external behaviors. 
        """

        for gal_type in self.gal_types:

            # Set the method used to return Monte Carlo realizations of per-halo gal_type abundance
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

            ### Now move on to galaxy profiles
            gal_prof_model = self.model_blueprint[gal_type]['profile']

            # Create a new method for each galaxy profile parameter
            for gal_prof_param in gal_prof_model.gal_prof_func_dict.keys():
                new_method_name = gal_prof_param + '_' + gal_type
                new_method_behavior = partial(self._get_gal_prof_param, 
                    gal_prof_param, gal_type)
                setattr(self, new_method_name, new_method_behavior)


            ### Create a method to assign positions to each gal_type
            new_method_name = 'pos_'+gal_type
            new_method_behavior = partial(self.mc_pos, gal_type = gal_type)
            setattr(self, new_method_name, new_method_behavior)


    def _get_gal_prof_param(self, gal_prof_param, gal_type, *args, **kwargs):
        """ Private method used by `_set_primary_behaviors` to assign (possibly biased) 
        profile parameters to mock galaxies. 

        Parameters 
        ----------
        gal_prof_param : string 
            Name of the galaxy profile parameter. 
            Must be equal to one of the galaxy profile parameter names.
            For example, if the input ``gal_type`` pertains to 
            a satellite-like population tracing a (possibly biased) NFW profile, 
            then ``gal_prof_param`` would be ``gal_NFWmodel_conc``. 

        gal_type : string 
            Name of the galaxy population. 

        prim_haloprop : optional positional argument 

        sec_haloprop : optional positional argument 

        mock_galaxies : optional keyword argument 

        """

        gal_prof_model = self.model_blueprint[gal_type]['profile']
        
        if gal_prof_param not in gal_prof_model.gal_prof_func_dict.keys():

            if 'mock_galaxies' in kwargs.keys():
                gal_type_slice = kwargs['mock_galaxies']._gal_type_indices[gal_type]
                halo_prof_param_key = (model_defaults.host_haloprop_prefix + 
                    gal_prof_param[model_defaults[len(galprop_prefix):]]
                    )
                return getattr(kwargs['mock_galaxies'], halo_prof_param_key)[gal_type_slice]
            else:
                return None

        else:

            halo_prop_list = self.retrieve_relevant_haloprops(gal_type, *args, **kwargs)
            return gal_prof_model.gal_prof_func_dict[gal_prof_param](*halo_prop_list)


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


    @property 
    def halo_prof_func_dict(self):
        """ Method to derive the halo profile parameter function dictionary 
        from a collection of component models. 

        Notes 
        -----
        If there are multiple instances of the same underlying 
        halo profile model, a profile function is chosen essentially at random. 
        This is innocuous, since the multiple instances have already been ensured 
        to provide consistent profile parameter functions. 

        """
        output_halo_prof_func_dict = {}

        for gal_type in self.gal_types:
            halo_prof_model = self.model_blueprint[gal_type]['profile'].halo_prof_model

            for key, func in halo_prof_model.halo_prof_func_dict.iteritems():
                output_halo_prof_func_dict[key] = func

        return output_halo_prof_func_dict


    def build_halo_prof_lookup_tables(self, prof_param_table_dict={}):

        for gal_type in self.gal_types:
            halo_prof_model = self.model_blueprint[gal_type]['profile'].halo_prof_model
            halo_prof_model.build_inv_cumu_lookup_table(prof_param_table_dict)

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
    """ Method searches a model_blueprint for all halo properties used by 
    any model, and returns a standard-form dictionary of the results. 

    Parameters 
    ----------
    model_blueprint : dict 
        Blueprint used by `~halotools.empirical_models.HodModelFactory` to 
        build a composite model. 

    Returns 
    -------
    output_dict : dict 
        Keys are `prim_haloprop_key`, `sec_haloprop_key` and `halo_boundary`. 
        Values are the strings giving the corresponding key in the halo catalog 
        containing the relevant data. 

    Notes 
    -----
    Used to set composite-model-wide values for the primary and secondary halo properties. 

    """

    prim_haloprop_list = []
    sec_haloprop_list = []
    halo_boundary_list = []
    
    no_prim_haloprop_msg = "For gal_type %s and feature %s, no primary haloprop detected"
    no_halo_boundary_msg = "For gal_type %s, no primary haloprop detected for profile model"

    # Build a list of all halo properties used by any component model, 
    # issuing warnings where necessary
    for gal_type in model_blueprint.keys():
        for feature in model_blueprint[gal_type].values():

            if 'prim_haloprop_key' in feature.haloprop_key_dict.keys():
                prim_haloprop_list.append(feature.haloprop_key_dict['prim_haloprop_key'])

            if 'sec_haloprop_key' in feature.haloprop_key_dict.keys():
                sec_haloprop_list.append(feature.haloprop_key_dict['sec_haloprop_key'])

            if 'halo_boundary' in feature.haloprop_key_dict.keys():
                halo_boundary_list.append(feature.haloprop_key_dict['halo_boundary'])

    # Run a bunch of tests on the each list of halo properties
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






























