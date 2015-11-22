# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a simulation halocat 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np
from multiprocessing import cpu_count
from copy import copy 
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
from astropy.table import Table 

from .mock_factory_template import MockFactory
from .mock_helpers import three_dim_pos_bundle, infer_mask_from_kwargs

from .. import model_helpers, model_defaults

try:
    from ... import mock_observables
    HAS_MOCKOBS = True
except ImportError:
    HAS_MOCKOBS = False

from ...sim_manager import sim_defaults
from ...utils.array_utils import randomly_downsample_data
from ...utils.table_utils import SampleSelector
from ...sim_manager import FakeSim, FakeMock
from ...custom_exceptions import *


__all__ = ['HodMockFactory']
__author__ = ['Andrew Hearin']


class HodMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies based on an HOD-style model. 

    Can be thought of as a factory that takes a model  
    and simulation halocat as input, 
    and generates a mock galaxy population. 
    The returned collection of galaxies possesses whatever 
    attributes were requested by the model, such as xyz position,  
    central/satellite designation, star-formation rate, etc. 

    """

    def __init__(self, populate=True, **kwargs):
        """
        Parameters 
        ----------
        halocat : object, keyword argument
            Object containing the halo catalog and other associated data.  
            Produced by `~halotools.sim_manager.supported_sims.HaloCatalog`

        model : object, keyword argument
            A model built by a sub-class of `~halotools.empirical_models.HodModelFactory`. 

        additional_haloprops : string or list of strings, optional   
            Each entry in this list must be a column key of ``halocat.halo_table``. 
            For each entry of ``additional_haloprops``, each member of 
            `mock.galaxy_table` will have a column key storing this property of its host halo. 
            If ``additional_haloprops`` is set to the string value ``all``, 
            the galaxy table will inherit every halo property in the catalog. Default is None. 

        populate : boolean, optional   
            If set to ``False``, the class will perform all pre-processing tasks 
            but will not call the ``model`` to populate the ``galaxy_table`` 
            with mock galaxies and their observable properties. Default is ``True``. 

        apply_completeness_cut : bool, optional 
            If True, only halos passing the mass completeness cut defined in 
            `~halotools.empirical_models.model_defaults` will be used to populate the mock. 
            Default is True. 
        """

        super(HodMockFactory, self).__init__(populate=populate, **kwargs)

        self.preprocess_halo_catalog()

        if populate is True:
            self.populate()

    def preprocess_halo_catalog(self, apply_completeness_cut = True, **kwargs):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. This pre-processing includes identifying the 
        catalog columns that will be used by the model to create the mock, 
        building lookup tables associated with the halo profile, 
        and possibly creating new halo properties. 

        Parameters 
        ----------
        logrmin : float, optional 
            Minimum radius used to build the lookup table for the halo profile. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        logrmax : float, optional 
            Maximum radius used to build the lookup table for the halo profile. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        Npts_radius_table : int, optional 
            Number of control points used in the lookup table for the halo profile.
            Default is set in `~halotools.empirical_models.model_defaults`. 

        apply_completeness_cut : bool, optional 
            If True, only halos passing the mass completeness cut defined in 
            `~halotools.empirical_models.model_defaults` will be used to populate the mock. 
            Default is True. 
        """

        ################ Make cuts on halo catalog ################
        # Select host halos only, since this is an HOD-style model
        self.halo_table = SampleSelector.host_halo_selection(
            table = self.halo_table)

        # make a conservative mvir completeness cut 
        # This cut can be controlled by changing sim_defaults.Num_ptcl_requirement
        if apply_completeness_cut is True:
            cutoff_mvir = sim_defaults.Num_ptcl_requirement*self.halocat.particle_mass
            mass_cut = (self.halo_table['halo_mvir'] > cutoff_mvir)
            self.halo_table = self.halo_table[mass_cut]

        ############################################################

        ### Create new columns of the halo catalog, if applicable
        try:
            d = self.model.new_haloprop_func_dict
            for new_haloprop_key, new_haloprop_func in d.iteritems():
                self.halo_table[new_haloprop_key] = new_haloprop_func(table = self.halo_table)
                self.additional_haloprops.append(new_haloprop_key)
        except AttributeError:
            pass

        self.model.build_lookup_tables(**kwargs)

    def populate(self, **kwargs):
        """ Method populating halos with mock galaxies. 
        """
        self.allocate_memory()

        # Loop over all gal_types in the model 
        for gal_type in self.gal_types:

            # Retrieve the indices of our pre-allocated arrays 
            # that store the info pertaining to gal_type galaxies
            gal_type_slice = self._gal_type_indices[gal_type]
            # gal_type_slice is a slice object

            # For the gal_type_slice indices of 
            # the pre-allocated array self.gal_type, 
            # set each string-type entry equal to the gal_type string
            self.galaxy_table['gal_type'][gal_type_slice] = (
                np.repeat(gal_type, self._total_abundance[gal_type],axis=0))

            # Store all other relevant host halo properties into their 
            # appropriate pre-allocated array 
            for halocatkey in self.additional_haloprops:
                self.galaxy_table[halocatkey][gal_type_slice] = np.repeat(
                    self.halo_table[halocatkey], self._occupation[gal_type], axis=0)

        self.galaxy_table['x'] = self.galaxy_table['halo_x']
        self.galaxy_table['y'] = self.galaxy_table['halo_y']
        self.galaxy_table['z'] = self.galaxy_table['halo_z']
        self.galaxy_table['vx'] = self.galaxy_table['halo_vx']
        self.galaxy_table['vy'] = self.galaxy_table['halo_vy']
        self.galaxy_table['vz'] = self.galaxy_table['halo_vz']

        for method in self._remaining_methods_to_call:
            func = getattr(self.model, method)
            gal_type_slice = self._gal_type_indices[func.gal_type]
            func(table = self.galaxy_table[gal_type_slice])
                
        # Positions are now assigned to all populations. 
        # Now enforce the periodic boundary conditions for all populations at once
        self.galaxy_table['x'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['x'], self.halocat.Lbox)
        self.galaxy_table['y'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['y'], self.halocat.Lbox)
        self.galaxy_table['z'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['z'], self.halocat.Lbox)

        if hasattr(self.model, 'galaxy_selection_func'):
            mask = self.model.galaxy_selection_func(self.galaxy_table)
            self.galaxy_table = self.galaxy_table[mask]

    def allocate_memory(self):
        """ Method allocates the memory for all the numpy arrays 
        that will store the information about the mock. 
        These arrays are bound directly to the mock object. 

        The main bookkeeping devices generated by this method are 
        ``_occupation`` and ``_gal_type_indices``. 

        """

        self.galaxy_table = Table() 

        # We will keep track of the calling sequence with a list called _remaining_methods_to_call
        # Each time a function in this list is called, we will remove that function from the list
        # Mock generation will be complete when _remaining_methods_to_call is exhausted
        self._remaining_methods_to_call = copy(self.model._mock_generation_calling_sequence)

        # Call all composite model methods that should be called prior to mc_occupation 
        # All such function calls must be applied to the table, since we do not yet know 
        # how much memory we need for the mock galaxy_table
        galprops_assigned_to_halo_table = []
        for func_name in self.model._mock_generation_calling_sequence:
            if 'mc_occupation' in func_name:
                break
            else:
                func = getattr(self.model, func_name)
                func(table = self.halo_table)
                galprops_assigned_to_halo_table_by_func = func._galprop_dtypes_to_allocate.names
                galprops_assigned_to_halo_table.extend(galprops_assigned_to_halo_table_by_func)
                self._remaining_methods_to_call.remove(func_name)
        # Now update the list of additional_haloprops, if applicable
        # This is necessary because each of the above function calls created new 
        # columns for the *halo_table*, not the *galaxy_table*. So we will need to use 
        # np.repeat inside mock.populate() so that mock galaxies inherit these newly-created columns
        # Since there is already a loop over additional_haloprops inside mock.populate() that does this, 
        # then all we need to do is append to this list
        galprops_assigned_to_halo_table = list(set(
            galprops_assigned_to_halo_table))
        self.additional_haloprops.extend(galprops_assigned_to_halo_table)
        self.additional_haloprops = list(set(self.additional_haloprops))

        self._occupation = {}
        self._total_abundance = {}
        self._gal_type_indices = {}

        first_galaxy_index = 0
        for gal_type in self.gal_types:
            occupation_func_name = 'mc_occupation_'+gal_type
            occupation_func = getattr(self.model, occupation_func_name)
            # Call the component model to get a Monte Carlo
            # realization of the abundance of gal_type galaxies
            self._occupation[gal_type] = occupation_func(table=self.halo_table)

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
            # Remove the mc_occupation function from the list of methods to call
            self._remaining_methods_to_call.remove(occupation_func_name)
            galprops_assigned_to_halo_table_by_func = occupation_func._galprop_dtypes_to_allocate.names
            self.additional_haloprops.extend(galprops_assigned_to_halo_table_by_func)
            
        self.Ngals = np.sum(self._total_abundance.values())

        # Allocate memory for all additional halo properties, 
        # including profile parameters of the halos such as 'conc_NFWmodel'
        for halocatkey in self.additional_haloprops:
            self.galaxy_table[halocatkey] = np.zeros(self.Ngals, 
                dtype = self.halo_table[halocatkey].dtype)

        # Separately allocate memory for the galaxy profile parameters
        for galcatkey in self.model.prof_param_keys:
            self.galaxy_table[galcatkey] = 0.

        self.galaxy_table['gal_type'] = np.zeros(self.Ngals, dtype=object)

        dt = self.model._galprop_dtypes_to_allocate
        for key in dt.names:
            self.galaxy_table[key] = np.zeros(self.Ngals, dtype = dt[key].type)

