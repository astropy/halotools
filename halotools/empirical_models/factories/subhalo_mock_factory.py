# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a simulation snapshot 
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
from ...sim_manager import FakeSim, FakeMock
from ...custom_exceptions import *


__all__ = ['SubhaloMockFactory']
__author__ = ['Andrew Hearin']



class SubhaloMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies.

    """

    def __init__(self, populate=True, **kwargs):
        """
        Parameters 
        ----------
        snapshot : object, keyword argument 
            Object containing the halo catalog and other associated data.  
            Produced by `~halotools.sim_manager.supported_sims.HaloCatalog`

        model : object, keyword argument
            A model built by a sub-class of `~halotools.empirical_models.SubhaloModelFactory`. 

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

        super(SubhaloMockFactory, self).__init__(populate=populate, **kwargs)

        # Pre-compute any additional halo properties required by the model
        self.preprocess_halo_catalog()
        self.precompute_galprops()

        if populate is True:
            self.populate()

    def preprocess_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. 
        """

        # Make any cuts on the halo catalog requested by the composite model
        if hasattr(self.model, 'halocut_funcobj'):
            self.halo_table = self.model.halocut_funcobj(halo_table=self.halo_table)

        ### Create new columns of the halo catalog, if applicable
        if hasattr(self.model, 'new_haloprop_func_dict'):
            for new_haloprop_key, new_haloprop_func in self.model.new_haloprop_func_dict.iteritems():
                self.halo_table[new_haloprop_key] = new_haloprop_func(halo_table=self.halo_table)
                self.additional_haloprops.append(new_haloprop_key)


    def precompute_galprops(self):
        """ Method pre-processes the input subhalo catalog, and pre-computes 
        all halo properties that will be inherited by the ``galaxy_table``. 
        """

        for key in self.additional_haloprops:
            self.galaxy_table[key] = self.halo_table[key]

        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for newkey in phase_space_keys:
            self.galaxy_table[newkey] = self.galaxy_table[model_defaults.host_haloprop_prefix+newkey]

        self.galaxy_table['galid'] = np.arange(len(self.galaxy_table))

        for galprop in self.model.galprop_list:
            component_model = self.model.model_blueprint[galprop]
            if hasattr(component_model, 'gal_type_func'):
                newkey = galprop + '_gal_type'
                self.galaxy_table[newkey] = (
                    component_model.gal_type_func(halo_table=self.galaxy_table)
                    )

    def populate(self):
        """ Method populating subhalos with mock galaxies. 
        """
        for galprop_key in self.model.galprop_list:
            
            model_func_name = 'mc_'+galprop_key
            model_func = getattr(self.model, model_func_name)
            self.galaxy_table[galprop_key] = model_func(halo_table=self.galaxy_table)

        if hasattr(self.model, 'galaxy_selection_func'):
            mask = self.model.galaxy_selection_func(self.galaxy_table)
            self.galaxy_table = self.galaxy_table[mask]







