# -*- coding: utf-8 -*-
"""
Module used to construct mock galaxy populations 
based on models that populate subhalos. 

"""

import numpy as np
from copy import copy 

from .mock_factory_template import MockFactory

from .. import model_helpers, model_defaults
from ...custom_exceptions import *


__all__ = ['SubhaloMockFactory']
__author__ = ['Andrew Hearin']


class SubhaloMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies based on models generated by 
    `~halotools.empirical_models.SubhaloModelFactory`. 

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

        additional_haloprops : string or list of strings, optional   
            Each entry in this list must be a column key of ``snapshot.halo_table``. 
            For each entry of ``additional_haloprops``, each member of 
            `mock.galaxy_table` will have a column key storing this property of its host halo. 
            If ``additional_haloprops`` is set to the string value ``all``, 
            the galaxy table will inherit every halo property in the catalog. Default is None. 

        populate : boolean, optional   
            If set to ``False``, the class will perform all pre-processing tasks 
            but will not call the ``model`` to populate the ``galaxy_table`` 
            with mock galaxies and their observable properties. Default is ``True``. 
        """

        super(SubhaloMockFactory, self).__init__(populate = populate, **kwargs)

        # Pre-compute any additional halo properties required by the model
        self.preprocess_halo_catalog()
        self.precompute_galprops()

        if populate is True:
            self.populate()

    def preprocess_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. 
        """

        ### Create new columns of the halo catalog, if applicable
        try:
            d = self.model.new_haloprop_func_dict
            for new_haloprop_key, new_haloprop_func in d.iteritems():
                self.halo_table[new_haloprop_key] = new_haloprop_func(halo_table = self.halo_table)
                self.additional_haloprops.append(new_haloprop_key)
        except AttributeError:
            pass


    def precompute_galprops(self):
        """ Method pre-processes the input subhalo catalog, and pre-computes 
        all halo properties that will be inherited by the ``galaxy_table``. 

        For example, in subhalo-based models, the phase space coordinates of the 
        galaxies are hard-wired to be equal to the phase space coordinates of the 
        parent subhalos, so these keys of the galaxy_table 
        can be pre-computed once and for all. 

        Additionally, a feature of some composite models may have explicit dependence 
        upon the type of halo/galaxy. The `gal_type_func` mechanism addresses this potential need 
        by adding an additional column(s) to the galaxy_table. These additional columns 
        can also be pre-computed as halo types do not depend upon model parameter values. 
        """

        self._precomputed_galprop_list = []

        for key in self.additional_haloprops:
            self.galaxy_table[key] = self.halo_table[key]
            self._precomputed_galprop_list.append(key)

        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for newkey in phase_space_keys:
            self.galaxy_table[newkey] = self.galaxy_table[model_defaults.host_haloprop_prefix+newkey]
            self._precomputed_galprop_list.append(newkey)

        self.galaxy_table['galid'] = np.arange(len(self.galaxy_table))
        self._precomputed_galprop_list.append('galid')

        for feature, component_model in self.model.model_blueprint.iteritems():

            try:
                f = component_model.gal_type_func
                newkey = feature + '_gal_type'
                self.galaxy_table[newkey] = f(halo_table=self.galaxy_table)
                self._precomputed_galprop_list.append(newkey)
            except AttributeError:
                pass
            except:
                clname = component_model.__class__.__name__
                msg = ("\nThe `gal_type_func` attribute of the " + clname + 
                    "\nraises an unexpected exception when passed a halo table as a "
                    "halo_table keyword argument. \n"
                    "If the features in your component model have explicit dependence "
                    "on galaxy type, \nthen you must implement the `gal_type_func` mechanism "
                    "in such a way that\nthis function accepts a "
                    "length-N halo table as a ``halo_table`` keyword argument, \n"
                    "and returns a length-N array of strings.\n")
                raise HalotoolsError(msg)

    def populate(self):
        """ Method populating subhalos with mock galaxies. 
        """
        self._allocate_memory()

        for method in self.model._mock_generation_calling_sequence:
            func = getattr(self.model, method)
            func(halo_table = self.galaxy_table)

        if hasattr(self.model, 'galaxy_selection_func'):
            mask = self.model.galaxy_selection_func(self.galaxy_table)
            self.galaxy_table = self.galaxy_table[mask]

    def _allocate_memory(self):
        """
        """
        Ngals = len(self.galaxy_table)

        # Allocate or overwrite any non-static galaxy propery 
        for key in self.model._galprop_dtypes_to_allocate.names:
            if key not in self._precomputed_galprop_list:
                dt = self.model._galprop_dtypes_to_allocate[key]
                self.galaxy_table[key] = np.empty(Ngals, dtype = dt)








