# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a simulation snapshot 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np
import occupation_helpers as occuhelp
import model_defaults
from ..sim_manager import sim_defaults

__all__ = ["HodMockFactory"]

class HodMockFactory(object):
    """ The constructor of this class takes 
    a snapshot and a composite model as input, 
    and returns a Monte Carlo realization of the composite model 
    painted onto the input snapshot. 
    """

    def __init__(self, snapshot, composite_model, 
        bundle_into_table=True, populate=True,
        additional_haloprops=[], new_haloprop_func_dict={}):

        # Bind the inputs to the mock object
        self.snapshot = snapshot
        self.halos = snapshot.halos
        self.particles = snapshot.particles
        self.model = composite_model

        self.additional_haloprops = additional_haloprops
        # Make sure all the default haloprops are included
        self.additional_haloprops.extend(model_defaults.haloprop_list) 
        # Remove any possibly redundant items
        self.additional_haloprops = list(set(self.additional_haloprops))

        self.new_haloprop_func_dict = new_haloprop_func_dict

        self.gal_types, self._occupation_bounds = self._get_gal_types()

        # Pre-compute any additional halo properties required by the model, 
        # such as 'NFWmodel_conc'. Also build all necessary lookup tables.
        self.process_halo_catalog()

        self._is_allocated = False

        if populate==True: self.populate()


    def process_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. This processing includes identifying the 
        catalog columns that will be used to construct the mock, 
        and building lookup tables associated with the halo profile. 
        """

        #############
        #### Make cuts on halo catalog
        # select host halos only
        host_halo_cut = (self.halos['upid']==-1)
        self.halos = self.halos[host_halo_cut]
        # make mvir completeness cut
        cutoff_mvir = sim_defaults.Num_ptcl_requirement*self.snapshot.particle_mass
        mass_cut = (self.halos['mvir'] > cutoff_mvir)
        self.halos = self.halos[mass_cut]
        #############

        for new_haloprop_key, new_haloprop_func in self.new_haloprop_func_dict.iteritems():
            self.halos[new_haloprop_key] = new_haloprop_func(self.halos)
            self.additional_haloprops.append(new_haloprop_key)

        self.prim_haloprop_key = self.model.prim_haloprop_key
        if hasattr(self.model,'sec_haloprop_key'): 
            self.sec_haloprop_key = self.model.sec_haloprop_key

        # Create new halo catalog columns associated with each 
        # parameter of the halo profile model, e.g., 'NFWmodel_conc'. 
        # The names of the new columns are the keys of the 
        # composite model's halo_prof_func_dict dictionary; 
        # each key's value is the function object that operates 
        # on the halos to create the new columns 
        function_dict = self.model.halo_prof_func_dict
        for new_haloprop_key, prof_param_func in function_dict.iteritems():
            self.halos[new_haloprop_key] = prof_param_func(self.halos[self.prim_haloprop_key])
            self.additional_haloprops.append(new_haloprop_key)

        # Create a list of galaxy profile parameter keys, 
        # one for each halo profile parameter key
        self.model._set_gal_prof_params()

        self.build_halo_prof_lookup_tables()


    def build_halo_prof_lookup_tables(self, input_prof_param_table_dict={}):

       # Compute the halo profile lookup table, ensuring that the min/max 
       # range spanned by the halo catalog is covered. The grid of parameters 
       # is defined by a tuple (xlow, xhigh, dx) in prof_param_table_dict, 
       # whose keys are the name of the halo profile parameter being discretized
        prof_param_table_dict={}
        if input_prof_param_table_dict == {}:
            for key in self.model.halo_prof_func_dict.keys():
                dpar = self.model.prof_param_table_dict[key][2]
                halocat_parmin = self.halos[key].min() - dpar
                model_parmin = self.model.prof_param_table_dict[key][0]
                parmin = np.min([halocat_parmin,model_parmin])
                halocat_parmax = self.halos[key].max() + dpar
                model_parmax = self.model.prof_param_table_dict[key][1]
                parmax = np.max([halocat_parmax,model_parmax])
                prof_param_table_dict[key] = (parmin, parmax, dpar)

        # Now over-write prof_param_table_dict with 
        # input_prof_param_table_dict, if applicable
        for key, value in input_prof_param_table_dict.iteritems():
            prof_param_table_dict[key] = value

        # Calling the following method will create new attributes of
        # self.model that can be used to discretize halo profiles.
        # Taking NFWProfile class as an example, the line of code that follows 
        # will create two new attributes of self.model:
        # 1. cumu_inv_param_table, an array of concentration bin boundaries, and 
        # 2. cumu_inv_func_table, an array of profile function objects, 
        # one function for each element of cumu_inv_param_table
        self.model.build_inv_cumu_lookup_table(
            prof_param_table_dict=prof_param_table_dict)


    def _get_gal_types(self):
        """ Assumes the gal_type list bound to the model has already been 
        sorted according to the occupation bounds. 
        """

        sorted_gal_type_list = self.model.gal_types

        occupation_bound = np.array([self.model.occupation_bound[gal_type] 
            for gal_type in self.model.gal_types])

        return sorted_gal_type_list, occupation_bound

    def populate(self):
        """ Method used to call the composite models to 
        sprinkle mock galaxies into halos. 
        """

        if self._is_allocated == False:
            self._allocate_memory()

        # Loop over all gal_types in the model 
        for gal_type in self.gal_types:
            # Retrieve the indices of our pre-allocated arrays 
            # that store the info pertaining to gal_type galaxies
            gal_type_slice = self._gal_type_indices[gal_type]

            # Set the value of the gal_type string
            getattr(self, 'gal_type')[gal_type_slice] = np.repeat(gal_type, 
                self._total_abundance[gal_type],axis=0)

            # Set the value of the primary halo property
            getattr(self, 'prim_haloprop_key')[gal_type_slice] = np.repeat(
                self.halos[self.model.prim_haloprop_key], 
                self._occupation[gal_type],axis=0)

            # Set the value of the secondary halo property, if relevant
            if hasattr(self.model, 'sec_haloprop_key'):
                getattr(self, 'sec_haloprop_key')[gal_type_slice] = np.repeat(
                    self.halos[self.model.sec_haloprop_key], 
                    self._occupation[gal_type],axis=0)

            # Bind all relevant host halo properties to the mock
            for halocatkey in self.additional_haloprops:
                galcatkey = model_defaults.host_haloprop_prefix+halocatkey
                getattr(self, galcatkey)[gal_type_slice] = np.repeat(
                    self.halos[halocatkey], self._occupation[gal_type], axis=0)

            # Call the SFR model, if relevant for this model
            if hasattr(self.model, 'sfr_model'):
                # Code this up later
                pass

            # Call the galaxy profile components
            for gal_prof_param_key in self.model.gal_prof_param_keys:
                getattr(self, gal_prof_param_key)[gal_type_slice] = (
                    self.model.gal_prof_param(gal_type, gal_prof_param_key, self)
                    )

            # Assign positions 
            # This function is called differently than other galaxy properties, 
            # since 'pos' is an attribute of any galaxy-halo model
            # and any gal_type, without exception
            pos_method_name = 'pos_'+gal_type
            getattr(self, 'pos')[gal_type_slice] = (
                getattr(self.model, pos_method_name)(self)
                )

            # Assign velocities, if relevant for this model
            if hasattr(self.model, 'vel'):
                pass

        # Positions are now assigned to all populations. 
        # Now enforce the periodic boundary conditions for all populations at once
        self.pos = occuhelp.enforce_periodicity_of_box(
            self.pos, self.snapshot.Lbox)

    def _allocate_memory(self):
        """ Method determines how many galaxies of each type 
        will populate the mock realization, initializes 
        various arrays to store mock catalog data, 
        and creates internal self._gal_type_indices attribute 
        for bookkeeping purposes. 
        """
        self._occupation = {}
        self._total_abundance = {}
        self._gal_type_indices = {}
        first_galaxy_index = 0
        for gal_type in self.gal_types:
            #print("Working on gal_type %s" % gal_type)
            #
            occupation_func_name = 'mc_occupation_'+gal_type
            occupation_func = getattr(self.model, occupation_func_name)
            # Call the component model to get a MC 
            # realization of the abundance of gal_type galaxies
            self._occupation[gal_type] = occupation_func(halos=self.halos)

            # Now use the above result to set up the indexing scheme
            self._total_abundance[gal_type] = (
                self._occupation[gal_type].sum()
                )
            last_galaxy_index = first_galaxy_index + self._total_abundance[gal_type]
            # Build a bookkeeping device to keep track of 
            # which array elements pertain to which gal_type. 
            self._gal_type_indices[gal_type] = slice(
                first_galaxy_index, last_galaxy_index)
            first_galaxy_index = last_galaxy_index

        self.Ngals = np.sum(self._total_abundance.values())
        self._is_allocated = True

        def _allocate_ndarray_attr(self, propname, example_entry):
            """ Private method of _allocate_memory used to create an empty 
            ndarray of the appropriate shape and bind it to the mock instance. 

            Parameters 
            ----------
            propname : string 
                Used to define the name of the attribute being created. 

            example_entry : array_like 
                Used to define the shape of attribute
            """
            if (type(example_entry) != str) & (type(example_entry) != np.string_):
                example_type = type(np.array(example_entry).flatten()[0])
            else:
                example_type = object

            example_shape = list(np.shape(example_entry))
            example_shape.insert(0, self.Ngals)
            total_entries = np.product(example_shape)
            setattr(self, propname, 
                np.zeros(total_entries,dtype=example_type).reshape(example_shape))

        # Allocate memory for all additional halo properties, 
        # including profile parameters of the halos such as 'halo_NFWmodel_conc'
        for halocatkey in self.additional_haloprops:
            galpropkey = model_defaults.host_haloprop_prefix+halocatkey
            example_entry = self.halos[halocatkey][0]
            _allocate_ndarray_attr(self, galpropkey, example_entry)

        # Separately allocate memory for the values of the (possibly biased)
        # galaxy profile parameters such as 'gal_NFWmodel_conc'
        for galcatkey in self.model.gal_prof_param_keys:
            example_entry = 0.
            _allocate_ndarray_attr(self, galcatkey, example_entry)

        _allocate_ndarray_attr(self, 'gal_type', self.gal_types[0])
        _allocate_ndarray_attr(self, 'prim_haloprop_key', 0.)
        if hasattr(self.model,'sec_haloprop_key'):
            _allocate_ndarray_attr(self, 'sec_haloprop_key', 0.)

        _allocate_ndarray_attr(self, 'pos', [0.,0.,0.])






































