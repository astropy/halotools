# -*- coding: utf-8 -*-
"""
Module containing the `~halotools.empirical_models.HodMockFactory` class, 
the primary class used to construct mock galaxy populations 
based on HOD-style models. 

The `~halotools.empirical_models.HodMockFactory` class 
provides an abstract interface between halo catalogs 
and Halotools models. 
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
from ...sim_manager import FakeSim
from ...custom_exceptions import *


__all__ = ['HodMockFactory']
__author__ = ['Andrew Hearin']

unavailable_haloprop_msg = ("Your model requires that the ``%s`` key appear in the halo catalog,\n"
    "but this column is not available in the catalog you attempted to populate.\n")

missing_halo_upid_msg = ("All HOD-style models populate host halos with mock galaxies.\n"
    "The way Halotools distinguishes host halos from subhalos is via the ``halo_upid`` column,\n"
    "with halo_upid = -1 for host halos and !=-1 for subhalos.\n"
    "The halo catalog you passed to the HodMockFactory does not have the ``halo_upid`` column.\n")

class HodMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies based on an HOD-style model 
    built by the `~halotools.empirical_models.HodModelFactory` class. 

    Can be thought of as a factory that takes a model  
    and simulation halocat as input, 
    and generates a mock galaxy population. 
    The returned collection of galaxies possesses whatever 
    attributes were requested by the model, such as xyz position,  
    central/satellite designation, star-formation rate, etc. 

    """

    def __init__(self, Num_ptcl_requirement=sim_defaults.Num_ptcl_requirement, 
        halo_mass_column_key = 'halo_mvir', **kwargs):
        """
        Parameters 
        ----------
        halocat : object, keyword argument
            Object containing the halo catalog and other associated data.  
            Produced by `~halotools.sim_manager.CachedHaloCatalog`

        model : object, keyword argument
            A model built by a sub-class of `~halotools.empirical_models.HodModelFactory`. 

        populate : boolean, optional   
            If set to ``False``, the class will perform all pre-processing tasks 
            but will not call the ``model`` to populate the ``galaxy_table`` 
            with mock galaxies and their observable properties. Default is ``True``. 

        Num_ptcl_requirement : int, optional 
            Requirement on the number of dark matter particles in the halo. 
            The column defined by the ``halo_mass_column_key`` string will have a cut placed on it: 
            all halos with halocat.halo_table[halo_mass_column_key] < Num_ptcl_requirement*halocat.particle_mass
            will be thrown out immediately after reading the original halo catalog in memory. 
            Default value is set in `~halotools.sim_defaults.Num_ptcl_requirement`. 

        halo_mass_column_key : string, optional 
            This string must be a column of the input halo catalog. 
            The column defined by this string will have a cut placed on it: 
            all halos with halocat.halo_table[halo_mass_column_key] < Num_ptcl_requirement*halocat.particle_mass
            will be thrown out immediately after reading the original halo catalog in memory. 
            Default is 'halo_mvir'

        """

        MockFactory.__init__(self, **kwargs)

        halocat = kwargs['halocat']
        self.Num_ptcl_requirement = Num_ptcl_requirement
        self.halo_mass_column_key = halo_mass_column_key

        self.preprocess_halo_catalog(halocat)

    def preprocess_halo_catalog(self, halocat):
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

        """
        try:
            assert 'halo_upid' in list(halocat.halo_table.keys())
        except AssertionError:
            raise HalotoolsError(missing_halo_upid_msg)

        ################ Make cuts on halo catalog ################
        # Select host halos only, since this is an HOD-style model
        halo_table = SampleSelector.host_halo_selection(table = halocat.halo_table)

        # make a (possibly trivial) completeness cut 
        cutoff_mvir = self.Num_ptcl_requirement*self.particle_mass
        mass_cut = halo_table[self.halo_mass_column_key] > cutoff_mvir
        halo_table = halo_table[mass_cut]

        ############################################################

        ### Create new columns of the halo catalog, if applicable
        try:
            d = self.model.new_haloprop_func_dict
            for new_haloprop_key, new_haloprop_func in d.items():
                halo_table[new_haloprop_key] = new_haloprop_func(table = halo_table)
                self.additional_haloprops.append(new_haloprop_key)
        except AttributeError:
            pass

        self._orig_halo_table = Table()
        for key in self.additional_haloprops:
            try:
                self._orig_halo_table[key] = halo_table[key][:]
            except KeyError:
                raise HalotoolsError(unavailable_haloprop_msg % key)

        self.model.build_lookup_tables()

    def populate(self, **kwargs):
        """ 
        Method populating host halos with mock galaxies. 
        By calling the `populate` method of your mock, you will repopulate 
        the halo catalog with a new realization of the model based on 
        whatever values of the model parameters are currently stored in the 
        model ``param_dict``. 

        Parameters 
        ------------
        masking_function : function, optional 
            Function object used to place a mask on the halo table prior to 
            calling the mock generating functions. Calling signature of the 
            function should be to accept a single positional argument storing 
            a table, and returning a boolean numpy array that will be used 
            as a fancy indexing mask. All masked halos will be ignored during 
            mock population. Default is None. 

        enforce_PBC : bool, optional 
            If set to True, after galaxy positions are assigned the 
            `model_helpers.enforce_periodicity_of_box` will re-map 
            satellite galaxies whose positions spilled over the edge 
            of the periodic box. Default is True. This variable should only 
            ever be set to False when using the ``masking_function`` to 
            populate a specific spatial subvolume, as in that case PBCs 
            no longer apply. 

        Examples 
        ----------
        >>> from halotools.empirical_models import PrebuiltHodModelFactory
        >>> model_instance = PrebuiltHodModelFactory('zheng07')

        Here we will use a fake simulation, but you can populate mocks 
        using any instance of `~halotools.sim_manager.CachedHaloCatalog` or 
        `~halotools.sim_manager.UserSuppliedHaloCatalog`. 

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()
        >>> model_instance.populate_mock(halocat)

        Your ``model_instance`` now has a ``mock`` attribute bound to it. 
        You can call the `populate` method bound to the ``mock``, 
        which will repopulate the halo catalog with a new Monte Carlo 
        realization of the model. 

        >>> model_instance.mock.populate()

        If you want to change the behavior of your model, just change the 
        values stored in the ``param_dict``. Differences in the parameter values 
        will change the behavior of the mock-population. 

        >>> model_instance.param_dict['logMmin'] = 12.1
        >>> model_instance.mock.populate()

        """
        # The _testing_mode keyword is for unit-testing only 
        # it has been intentionally left out of the docstring
        try:
            self._testing_mode = kwargs['_testing_mode']
        except KeyError:
            self._testing_mode = False

        try:
            self.enforce_PBC = kwargs['enforce_PBC']
        except KeyError:
            self.enforce_PBC = True

        try:
            masking_function = kwargs['masking_function']
            mask = masking_function(self._orig_halo_table)
            self.halo_table = self._orig_halo_table[mask]
        except:
            self.halo_table = self._orig_halo_table

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
                
        if self.enforce_PBC is True:
            self.galaxy_table['x'], self.galaxy_table['vx'] = (
                model_helpers.enforce_periodicity_of_box(
                    self.galaxy_table['x'], self.Lbox, 
                    velocity = self.galaxy_table['vx'], 
                    check_multiple_box_lengths = self._testing_mode)
                )

            self.galaxy_table['y'], self.galaxy_table['vy'] = (
                model_helpers.enforce_periodicity_of_box(
                    self.galaxy_table['y'], self.Lbox, 
                    velocity = self.galaxy_table['vy'], 
                    check_multiple_box_lengths = self._testing_mode)
                )

            self.galaxy_table['z'], self.galaxy_table['vz'] = (
                model_helpers.enforce_periodicity_of_box(
                    self.galaxy_table['z'], self.Lbox, 
                    velocity = self.galaxy_table['vz'], 
                    check_multiple_box_lengths = self._testing_mode)
                )

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
            
        self.Ngals = np.sum(list(self._total_abundance.values()))

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

