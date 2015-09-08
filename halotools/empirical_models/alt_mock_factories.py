# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a simulation snapshot 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np
from multiprocessing import cpu_count

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from astropy.table import Table 

from . import model_helpers, model_defaults

from .mock_helpers import three_dim_pos_bundle, infer_mask_from_kwargs

from ..custom_exceptions import HalotoolsError
from ..mock_factories import MockFactory

try:
    from .. import mock_observables
    HAS_MOCKOBS = True
except ImportError:
    HAS_MOCKOBS = False

from ..sim_manager import sim_defaults
from ..utils.array_utils import randomly_downsample_data

from ..sim_manager import FakeSim, FakeMock


__all__ = ['AltHodMockFactory']
__author__ = ['Andrew Hearin']


class AltHodMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies based on an HOD-style model. 

    Can be thought of as a factory that takes a model  
    and simulation snapshot as input, 
    and generates a mock galaxy population. 
    The returned collection of galaxies possesses whatever 
    attributes were requested by the model, such as xyz position,  
    central/satellite designation, star-formation rate, etc. 

    """

    def __init__(self, populate=True, **kwargs):
        """
        Parameters 
        ----------
        snapshot : object, keyword argument
            Object containing the halo catalog and other associated data.  
            Produced by `~halotools.sim_manager.supported_sims.HaloCatalog`

        model : object, keyword argument
            A model built by a sub-class of `~halotools.empirical_models.HodModelFactory`. 

        additional_haloprops : list of strings, optional   
            Each entry in this list must be a column key of ``snapshot.halo_table``. 
            For each entry of ``additional_haloprops``, each member of the mock galaxy population 
            will have an attribute storing this property of its host halo. 
            The corresponding mock galaxy attribute name will be pre-pended by ``halo_``. 
            If ``additional_haloprops`` is set to the string value ``all``, 
            the galaxy table will inherit every halo property in the catalog. Default is None. 

        halocut_funcobj : function object, optional   
            Function object used to place a cut on the input ``snapshot.halo_table`` table. 
            Default behavior depends on the sub-class of `MockFactory`. 
            If the ``halocut_funcobj`` keyword argument is passed, 
            the input to the function must be a length-Nsubhalos structured numpy array or Astropy table; 
            the function output must be a length-Nsubhalos boolean array that will be used as a mask. 

        populate : boolean, optional   
            If set to ``False``, the class will perform all pre-processing tasks 
            but will not call the ``model`` to populate the ``galaxy_table`` 
            with mock galaxies and their observable properties. Default is ``True``. 
        """

        super(HodMockFactory, self).__init__(populate=populate, **kwargs)

        self.preprocess_halo_catalog()

        if populate is True:
            self.populate()

    def preprocess_halo_catalog(self, **kwargs):
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

        ################ Make cuts on halo catalog ################
        # Select host halos only, since this is an HOD-style model
        self.halo_table = self.snapshot.host_halos

        # make a conservative mvir completeness cut 
        # This can be relaxed by changing sim_defaults.Num_ptcl_requirement
        cutoff_mvir = sim_defaults.Num_ptcl_requirement*self.snapshot.particle_mass
        mass_cut = (self.halo_table['halo_mvir'] > cutoff_mvir)
        self.halo_table = self.halo_table[mass_cut]

        # Make any additional cuts requested by the composite model
        if hasattr(self.model, 'halocut_funcobj'):
            self.halo_table = self.model.halocut_funcobj(halo_table=self.halo_table)
        ############################################################

        ### Create new columns of the halo catalog, if applicable
        if hasattr(self.model, 'new_haloprop_func_dict'):
            for new_haloprop_key, new_haloprop_func in self.model.new_haloprop_func_dict.iteritems():
                self.halo_table[new_haloprop_key] = new_haloprop_func(halo_table=self.halo_table)
                self.additional_haloprops.append(new_haloprop_key)

        self.model.build_halo_prof_lookup_tables(**kwargs)

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


            # Assign phase space distribution 
            pos_method_name = 'pos_'+gal_type

            self.galaxy_table['x'][gal_type_slice], \
            self.galaxy_table['y'][gal_type_slice], \
            self.galaxy_table['z'][gal_type_slice] = (
                getattr(self.model, pos_method_name)(
                    halo_table=self.galaxy_table[gal_type_slice])
                )
                
        # Positions are now assigned to all populations. 
        # Now enforce the periodic boundary conditions for all populations at once
        self.galaxy_table['x'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['x'], self.snapshot.Lbox)
        self.galaxy_table['y'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['y'], self.snapshot.Lbox)
        self.galaxy_table['z'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['z'], self.snapshot.Lbox)

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

        self._occupation = {}
        self._total_abundance = {}
        self._gal_type_indices = {}

        first_galaxy_index = 0
        for gal_type in self.gal_types:
            occupation_func_name = 'mc_occupation_'+gal_type
            occupation_func = getattr(self.model, occupation_func_name)
            # Call the component model to get a Monte Carlo
            # realization of the abundance of gal_type galaxies
            self._occupation[gal_type] = occupation_func(halo_table=self.halo_table)

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

        # Allocate memory for all additional halo properties, 
        # including profile parameters of the halos such as 'NFWmodel_conc'
        for halocatkey in self.additional_haloprops:
            self.galaxy_table[halocatkey] = np.zeros(self.Ngals, 
                dtype = self.halo_table[halocatkey].dtype)

        # Separately allocate memory for the galaxy profile parameters
        for galcatkey in self.model.prof_param_keys:
            self.galaxy_table[galcatkey] = 0.

        self.galaxy_table['gal_type'] = np.zeros(self.Ngals, dtype=object)

        dt = self._galprop_dtypes_to_allocate
        for key in dt.names:
            self.galaxy_table[key] = np.zeros(self.Ngals, dtype = dt[key].type)


