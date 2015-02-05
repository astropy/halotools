# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a simulation snapshot 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np
import occupation_helpers as occuhelp
import defaults

class HodMockFactory(object):
    """ The constructor of this class takes 
    a snapshot and a composite model as input, 
    and returns a Monte Carlo realization of the composite model 
    painted onto the input snapshot. 
    """

    def __init__(self, snapshot, composite_model, 
        bundle_into_table=True, populate=True,
        additional_haloprops=[]):

        # Bind the inputs to the mock object
        self.snapshot = snapshot
        self.halos = snapshot.halos
        self.particles = snapshot.particles
        self.model = composite_model
        self.additional_haloprops = additional_haloprops

        # Bind a list of strings containing the gal_types 
        # of the composite model to the mock instance. 
        # The self.gal_types list is ordered such that 
        # populations with unity-bounded occupations appear first
        self.gal_types, self._occupation_bounds = self._get_gal_types()

        # There are various columns of the halo catalog 
        # to create in advance of populating a galaxy model. 
        # For example, a column called 'prim_haloprop' will be made 
        # in advance of populating any traditional empirical model. 
        # The composite model bound to the mock contains the instructions to give to 
        # the process_halo_catalog method about what columns to construct. 
        self.process_halo_catalog()

        if populate==True:
            self.populate()


    def process_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. This processing includes identifying the 
        catalog columns that will be used to construct the mock, 
        and building lookup tables associated with the halo profile. 
        """

        # Currently, all halo catalog columns added in this stage 
        # pertain only to those halo properties needed to generate 
        # halo profiles. However, it should be possible to add new columns 
        # that can be used as either the primary or secondary halo properties. 
        # And, additionally, different gal_types should be permitted to have 
        # different primary and secondary halo properties. 
        # Making these changes will require some new bookkeeping, including 
        # 1. having prim_haloprop_key and sec_haloprop_key be dictionaries 
        # with gal_type as keys, and 2. having the build_profile_lookup_tables 
        # method only create tables for the halo paramters pertaining to profiles, 
        # and ignoring new halo parameters. 

        self.prim_haloprop_key = self.model.prim_haloprop_key
        if hasattr(self.model,'sec_haloprop_key'): 
            self.sec_haloprop_key = self.model.sec_haloprop_key

        # Create new columns for self.halos associated with each 
        # parameter of the halo profile model, e.g., 'halo_NFW_conc'. 
        # The function objects that operate on the halos to create new columns 
        # are bound as values of the param_func_dict dictionary, whose keys 
        # are the column names to be created with those functions. 
        halo_prof_param_keys = []
        prim_haloprop = self.halos[self.prim_haloprop_key]
        function_dict = self.model.halo_prof_model.param_func_dict
        for key, prof_param_func in function_dict.iteritems():
            self.halos[key] = prof_param_func(prim_haloprop)
            halo_prof_param_keys.extend([key])
        # Create a convenient bookkeeping device to keep track of the 
        # halo profile parameter model keys that were added by the model
        setattr(self.halos, 'halo_prof_param_keys', halo_prof_param_keys)

        self.build_profile_lookup_tables()


    def build_profile_lookup_tables(self, prof_param_table_dict={}):

       # Compute the halo profile lookup table, ensuring that the min/max 
       # range spanned by the halo catalog is covered. The grid of parameters 
       # is defined by a tuple (xlow, xhigh, dx) in prof_param_table_dict, 
       # whose keys are the name of the halo profile parameter being digitized
        if prof_param_table_dict == {}:
            for key in self.halos.halo_prof_param_keys:
                dpar = self.model.halo_prof_model.prof_param_table_dict[key][2]
                halocat_parmin = self.halos[key].min() - dpar
                model_parmin = self.model.halo_prof_model.prof_param_table_dict[key][0]
                parmin = np.min(halocat_parmin,model_parmin)
                halocat_parmax = self.halos[key].max() + dpar
                model_parmax = self.model.halo_prof_model.prof_param_table_dict[key][1]
                parmax = np.max(halocat_parmax,model_parmax)
                prof_param_table_dict[key] = (parmin, parmax, dpar)

        # Calling the following method will create new attributes 
        # for self.model.halo_prof_model that can be used to discretize 
        # the galaxy profile parameters and the functions that govern the profile
        # Using NFWProfile class as an example, there will be two new attributes: 
        # 1. cumu_inv_conc_table, an array of concentration bins, and 
        # 2. cumu_inv_func_table, an array of profile function objects, 
        # one function for each element of cumu_inv_conc_table
        self.model.halo_prof_model.build_inv_cumu_lookup_table(
            prof_param_table_dict=prof_param_table_dict)


    def _get_gal_types(self, testmode=defaults.testmode):
        """ Internal bookkeeping method used to conveniently bind the gal_types of a 
        composite model, and their occupation bounds, to the mock object. 

        This method identifies all gal_type strings used in the composite model, 
        and creates an array of those strings, ordered such that gal_types with 
        unity-bounded occupations (e.g., centrals) appear first. 
        """

        occupation_bound = np.array([self.model.occupation_bound[gal_type] 
            for gal_type in self.model.gal_types])

        if testmode==True:
            if (set(occupation_bound) != {1, float("inf")}):
                raise ValueError("The only supported finite occupation bound is unity,"
                    " otherwise it must be set to infinity")

        sorted_idx = np.argsort(occupation_bound)
        occupation_bound = occupation_bound[sorted_idx]
        sorted_gal_type_list = self.model.gal_types[sorted_idx]

        return sorted_gal_type_list, occupation_bound


    def _set_mock_attributes(self, testmode=defaults.testmode):
        """ Internal method used to create self._mock_galprops and 
        self._mock_haloprops, which are lists of strings 
        of halo and galaxy properties 
        that will be bound to the mock object. 
        """

        # The entries of self._mock_haloprops (which are strings) 
        # refer to column names of the halo catalog upon which the mock is built. 
        # Each galaxy's host halo property listed in self._mock_haloprops will be included 
        # in the final data structure containing the collection of mock galaxies. 
        # However, in the output mock galaxy table, these column names will be 
        # prepended by host_haloprop_prefix, set in halotools.defaults
        _mock_haloprops = defaults.haloprop_list # store the strings in a temporary list
        _mock_haloprops.extend(self.additional_haloprops)

        # Now we use a conditional list comprehension to ensure 
        # that all entries begin with host_haloprop_prefix, 
        # and also that host_haloprop_prefix is not duplicated
        # This just protects against what should be considered an innocuous usage error
        prefix = defaults.host_haloprop_prefix
        self._mock_haloprops = (
            [entry if entry[0:len(prefix)]==prefix else prefix+entry for entry in _mock_haloprops]
            )

        # Some models require ...
        self._mock_halomodelprops = self.halos.halo_prof_param_keys
        self._mock_galmodelprops = self.model._example_attr_dict.keys()

        if testmode==True:
            assert len(self._mock_halomodelprops)==len(set(self._mock_halomodelprops))
            assert len(self._mock_galmodelprops)==len(set(self._mock_galmodelprops))

        # Throw away any possible repeated entries
        self._mock_haloprops = list(set(self._mock_haloprops))
        self._mock_halomodelprops = list(set(self._mock_halomodelprops))
        self._mock_galmodelprops = self.model._example_attr_dict.keys()


    def populate(self):
        """ Method used to call the composite models to 
        sprinkle mock galaxies into halos. 
        """

        self._allocate_memory()

        # Loop over all gal_types in the model 
        for gal_type in self.gal_types:
            # Retrieve the indices of our pre-allocated arrays 
            # that store the info pertaining to gal_type galaxies
            gal_type_slice = self._gal_type_indices[gal_type]

            # Set the value of the gal_type string
            self.gal_type[gal_type_slice] = np.repeat(gal_type, 
                self._occupation[gal_type].sum())

            # Set the value of the primary halo property
            self.prim_haloprop[gal_type_slice] = np.repeat(
                self.halos[self.prim_haloprop_key], 
                self._occupation[gal_type])

            # Set the value of the secondary halo property, if relevant
            if hasattr(self.model, 'sec_haloprop_key'):
                self.sec_haloprop[gal_type_slice] = np.repeat(
                    self.halos[self.sec_haloprop_key], 
                    self._occupation[gal_type])


            # Bind all additional halo properties to the mock
            for propname in self._mock_haloprops:
                # In the mock galaxy catalog, host halo properties have the same 
                # column names as in the halo catalog, but prepended by the halo prefix 
                # set in defaults module. 
                # So strip the halo prefix to access the columns of the halo catalog
                halocatkey = propname[len(defaults.host_haloprop_prefix):]
                getattr(self, propname)[gal_type_slice] = np.repeat(
                    self.halos[halocatkey], self._occupation[gal_type])

            # Each halo has had one or more profile model parameters attached to it 
            # during the call to process_halo_catalog(). Bind each of those 
            # profile parameter values to the mock galaxy. 
            # Note that these do not vary with the gal_type. 
            for propname in self._mock_halomodelprops:
                getattr(self, propname)[gal_type_slice] = np.repeat(
                    self.halos[propname], self._occupation[gal_type])

            # The following for loop does not work properly 
            for propname in self._mock_galmodelprops:
                if propname in self.model._gal_type_example_attr_dict.keys():
                    getattr(self, propname)[gal_type_slice] = (
                        self.model.retrieve_component_behavior(
                            self, propname, gal_type)
                        )

        # Positions are now assigned to all populations. 
        # Now enforce the periodic boundary conditions of the simulation box
        self.coords = occuhelp.enforce_periodicity_of_box(
            self.coords, self.snapshot.Lbox)

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
            # Call the component model to get a MC 
            # realization of the abundance of gal_type galaxies
            self._occupation[gal_type] = (
                self.model.mc_occupation(
                    gal_type, self.halos))
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
            example_shape = list(np.shape(example_entry))
            example_shape.insert(0, self.Ngals)
            total_entries = np.product(example_shape)
            setattr(self, propname, 
                np.zeros(total_entries).reshape(example_shape))

        for propname in self._mock_galprops:
            example_entry = defaults.galprop_dict[propname]
            _allocate_ndarray_attr(self, propname, example_entry)

        for propname in self._mock_haloprops:
            # for halo catalog-derived properties 
            # we need to strip the prefix from the string
            halocatkey = propname[len(defaults.host_haloprop_prefix):]
            example_entry = self.halos[halocatkey][0]
            _allocate_ndarray_attr(self, propname, example_entry)

        for propname in self._mock_halomodelprops:
            example_entry = 0 # All profile model properties, e.g., conc, are scalars
            _allocate_ndarray_attr(self, propname, example_entry)

        for propname in self._mock_galmodelprops:
            example_entry = self.model._example_attr_dict[propname]
            _allocate_ndarray_attr(self, propname, example_entry)

        _allocate_ndarray_attr(self, 'prim_haloprop', 0)
        _allocate_ndarray_attr(self, 'gal_type', 0)
        if hasattr(self.model,'sec_haloprop_key'):
            _allocate_ndarray_attr(self, 'sec_haloprop', 0)



































