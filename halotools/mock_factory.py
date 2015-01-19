# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a simulation snapshot 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np
import occupation_helpers as occuhelp

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
        self.gal_types = self._get_gal_types()

        # The process_halo_catalog method 
        self.process_halo_catalog()

        if populate==True:
            self.populate()


    def process_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. This processing includes identifying the 
        catalog columns that will be used to construct the mock, 
        and building lookup tables associated with the halo profile. 
        """

        self.prim_haloprop_key = self.model.prim_haloprop_key
        if hasattr(self.model,'sec_haloprop_key'): 
            self.sec_haloprop_key = self.model.sec_haloprop_key

        # Create new columns for self.halos associated with each 
        # parameter of the halo profile model, e.g., 'halo_NFW_conc'. 
        halo_prof_param_keys = []
        prim_haloprop = self.halos[self.prim_haloprop_key]
        halo_prof_dict = self.model.halo_prof_model.param_func_dict
        for key, prof_param_func in halo_prof_dict.iteritems():
            self.halos[key] = prof_param_func(prim_haloprop)
            halo_prof_param_keys.extend([key])
        # Create a convenient bookkeeping device to keep track of the 
        # halo profile parameter model keys
        setattr(self.halos, 'halo_prof_param_keys', halo_prof_param_keys)

        self.build_profile_lookup_tables()


    def build_profile_lookup_tables(self, prof_param_table_dict={}):

       # Compute the halo profile lookup table, ensuring that the min/max 
       # range spanned by the halo catalog is covered. 
        if prof_param_table_dict != {}:
            for key in self.halos.halo_prof_param_keys:
                dpar = self.model.halo_prof_model.prof_param_table_dict[key][2]
                halocat_parmin = self.halos[key].min() - dpar
                model_parmin = self.model.halo_prof_model.prof_param_table_dict[key][0]
                parmin = np.min(halocat_parmin,model_parmin)
                halocat_parmax = self.halos[key].max() + dpar
                model_parmax = self.model.halo_prof_model.prof_param_table_dict[key][1]
                parmax = np.max(halocat_parmax,model_parmax)
                prof_param_table_dict[key] = (parmin, parmax, dpar)

        self.model.halo_prof_model.build_inv_cumu_lookup_table(
            prof_param_table_dict=prof_param_table_dict)


    def _get_gal_types(self):
        """ Internal bookkeeping method used to conveniently bind the gal_types of a 
        composite model, and their occupation bounds, to the mock object. 

        This method identifies all gal_type strings used in the composite model, 
        and creates an array of those strings, ordered such that gal_types with 
        unit-bounded occupations (e.g., centrals) appear first. 
        """

        # Set the gal_types attribute, sorted so that bounded populations appear first
        self._occupation_bound = np.array([self.model.occupation_bound[gal_type] 
            for gal_type in self.model.gal_types])

        if defaults.testmode==True:
            if (set(self._occupation_bound) != {1, float("inf")}):
                raise ValueError("The only supported finite occupation bound is unity,"
                    " otherwise it must be set to infinity")

        sorted_idx = np.argsort(self._occupation_bound)
        self._occupation_bound = self._occupation_bound[sorted_idx]
        sorted_gal_type_list = self.model.gal_types[sorted_idx]

        return sorted_gal_type_list


    def _set_mock_attributes(self):
        """ Internal method used to create self._mock_galprops and 
        self._mock_haloprops, which are lists of strings 
        of halo and galaxy properties 
        that will be bound to the mock object. 
        """

        # The entries of _mock_galprops will be used as column names in the 
        # data structure containing the mock galaxies
        self._mock_galprops = defaults.galprop_dict.keys()

        # Currently the composite model is not set up to create this list
        self._mock_galprops.extend(self.model.additional_galprops)

        # Throw away any possible repeated entries
        self._mock_galprops = list(set(self._mock_galprops))

        # The entries of self._mock_haloprops (which are strings) 
        # will be used as column names in the 
        # data structure containing the mock galaxies, 
        # but prepended by host_haloprop_prefix, set in halotools.defaults
        _mock_haloprops = defaults.haloprop_list # store the strings in a temporary list
        _mock_haloprops.extend(self.halos.halo_prof_param_keys)
        _mock_haloprops.extend(self.additional_haloprops)
        # Now we use a conditional list comprehension to ensure 
        # that all entries begin with host_haloprop_prefix, 
        # and also that host_haloprop_prefix is not needlessly duplicated
        prefix = defaults.host_haloprop_prefix
        self._mock_haloprops = (
            [entry if entry[0:len(prefix)]==prefix else prefix+entry for entry in _mock_haloprops]
            )
        # Throw away any possible repeated entries
        self._mock_haloprops = list(set(self._mock_haloprops))


    def populate(self):
        """ Method used to call the composite models to 
        sprinkle mock galaxies into halos. 
        """

        self._allocate_memory()

        # Loop over all gal_types in the model 
        for gal_type in self.gal_types:
            # Retrieve via hash lookup the indices 
            # storing gal_type galaxy info in our pre-allocated arrays
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

            # Bind all relevant halo properties to the mock
            for haloprop in self._mock_haloprops:
                # Strip the halo prefix
                key = haloprop[len(defaults.host_haloprop_prefix):]
                getattr(self, haloprop)[gal_type_slice] = np.repeat(
                    self.halos[key], self._occupation[gal_type])

        # Now need to call the phase space models for position and velocity

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

        for galprop in self._mock_galprops:
            example_entry = defaults.galprop_dict[galprop]
            example_shape = list(np.shape(example_entry))
            total_entries_pergal = self.Ngals*np.product(example_shape)
            example_shape.insert(0, self.Ngals)
            setattr(self, galprop, 
                np.zeros(total_entries_pergal).reshape(example_shape))

        for haloprop in self._mock_haloprops:
            # Strip the prefix from the string
            key = haloprop[len(defaults.host_haloprop_prefix):]
            example_entry = self.halos[key]
            example_shape = list(np.shape(example_entry))
            total_entries_pergal = self.Ngals*np.product(example_shape)
            example_shape.insert(0, self.Ngals)
            setattr(self, haloprop, 
                np.zeros(total_entries_pergal).reshape(example_shape))

        self.prim_haloprop = np.zeros(self.Ngals)
        if hasattr(self.model,'sec_haloprop_key'):
            self.sec_haloprop = np.zeros(self.Ngals)

































