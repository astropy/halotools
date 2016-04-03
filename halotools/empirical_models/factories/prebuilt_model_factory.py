# -*- coding: utf-8 -*-
"""
Module storing the factories used to generate 
Halotools-provided composite models 
of the galaxy-halo connection. 
"""

__all__ = ['PrebuiltSubhaloModelFactory', 'PrebuiltHodModelFactory']
__author__ = ['Andrew Hearin']

import numpy as np
from copy import copy
from warnings import warn 
import collections 

from ..factories import SubhaloModelFactory, HodModelFactory

from .. import model_helpers
from .. import model_defaults 

from ...sim_manager import sim_defaults
from ...sim_manager import FakeSim
from ...utils.array_utils import custom_len
from ...custom_exceptions import *

class PrebuiltSubhaloModelFactory(SubhaloModelFactory):
    """ 
    Factory class providing instances of 
    `SubhaloModelFactory` models that come prebuilt with Halotools. 
    For documentation on the methods bound to `PrebuiltSubhaloModelFactory`, 
    see the docstring of `~halotools.empirical_models.SubhaloModelFactory`. 

    """
    prebuilt_model_nickname_list = ['behroozi10']

    def __init__(self, model_nickname, **kwargs):
        """
        Parameters
        ----------
        model_nickname : string 
            String used to select the appropriate prebuilt 
            model_dictionary that will be used to build the instance. 
            See the ``Examples`` below. The list of available options are 

            * 'behroozi10' (see :ref:`behroozi10_composite_model`)
            
            * 'smhm_binary_sfr' (see `~halotools.empirical_models.smhm_binary_sfr_model_dictionary`)

        galaxy_selection_func : function object, optional  
            Function object that imposes a cut on the mock galaxies. 
            Function should take a length-k Astropy table as a single positional argument, 
            and return a length-k numpy boolean array that will be 
            treated as a mask over the rows of the table. If not None, 
            the mask defined by ``galaxy_selection_func`` will be applied to the 
            ``galaxy_table`` after the table is generated by the `populate_mock` method. 
            Default is None.  

        halo_selection_func : function object, optional   
            Function object used to place a cut on the input ``table``. 
            If the ``halo_selection_func`` keyword argument is passed, 
            the input to the function must be a single positional argument storing a 
            length-N structured numpy array or Astropy table; 
            the function output must be a length-N boolean array that will be used as a mask. 
            Halos that are masked will be entirely neglected during mock population.

        Examples 
        ----------

        >>> model_instance = PrebuiltSubhaloModelFactory('behroozi10', redshift = 2)

        Passing in `behroozi10` as the ``model_nickname`` argument triggers the factory to 
        call the `~halotools.empirical_models.behroozi10_model_dictionary` 
        function. When doing so, the remaining arguments that 
        were passed to the `PrebuiltSubhaloModelFactory` 
        will in turn be passed on to 
        `~halotools.empirical_models.behroozi10_model_dictionary`. 

        Now that we have built an instance of a composite model, we can use it to 
        populate any simulation in the Halotools cache: 

        >>> from halotools.sim_manager import CachedHaloCatalog # doctest: +SKIP
        >>> halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 2) # doctest: +SKIP
        >>> model_instance.populate_mock(halocat) # doctest: +SKIP

        As described in the `~halotools.empirical_models.ModelFactory.populate_mock` 
        docstring, calling the ``populate_mock`` method creates a ``mock`` attribute 
        bound to your model_instance. After you initially populate a halo catalog 
        using the `populate_mock` method, you can repopulate 
        the halo catalog by calling the 
        `~halotools.empirical_models.MockFactory.populate` method bound to 
        ``model_instance.mock``. 
        """
        input_model_dictionary, supplementary_kwargs = (
            self._retrieve_prebuilt_model_dictionary(model_nickname, **kwargs)
            )

        super_class_kwargs = {}
        for key, value in input_model_dictionary.items():
            super_class_kwargs[key] = value
        for key, value in supplementary_kwargs.items():
            super_class_kwargs[key] = value

        SubhaloModelFactory.__init__(self, **super_class_kwargs)


    def _retrieve_prebuilt_model_dictionary(self, model_nickname, **constructor_kwargs):
        """
        """
        forbidden_constructor_kwargs = ('model_feature_calling_sequence')
        for kwarg in forbidden_constructor_kwargs:
            if kwarg in constructor_kwargs:
                msg = ("\nWhen using the HodModelFactory to build an instance of a prebuilt model,\n"
                    "do not pass a ``%s`` keyword argument to the SubhaloModelFactory constructor.\n"
                    "The appropriate source of this keyword is as part of a prebuilt model dictionary.\n")
                raise HalotoolsError(msg % kwarg)


        model_nickname = model_nickname.lower()

        if model_nickname == 'behroozi10':
            from ..composite_models import smhm_models
            dictionary_retriever = smhm_models.behroozi10_model_dictionary
        elif model_nickname == 'smhm_binary_sfr':
            from ..composite_models import sfr_models
            dictionary_retriever = sfr_models.smhm_binary_sfr_model_dictionary
        else:
            msg = ("\nThe ``%s`` model_nickname is not recognized by Halotools\n")
            raise HalotoolsError(msg)

        result = dictionary_retriever(**constructor_kwargs)
        if type(result) is dict:
            input_model_dictionary = result
            supplementary_kwargs = {}
            supplementary_kwargs['model_feature_calling_sequence'] = None 
        elif type(result) is tuple:
            input_model_dictionary = result[0]
            supplementary_kwargs = result[1]
        else:
            raise HalotoolsError("Unexpected result returned from ``%s``\n"
            "Should be either a single dictionary or a 2-element tuple of dictionaries\n"
             % dictionary_retriever.__name__)

        return input_model_dictionary, supplementary_kwargs


class PrebuiltHodModelFactory(HodModelFactory):
    """ 
    Factory class providing instances of 
    `HodModelFactory` models that come prebuilt with Halotools. 
    For documentation on the methods bound to `PrebuiltHodModelFactory`, 
    see the docstring of `~halotools.empirical_models.HodModelFactory`. 
    """

    prebuilt_model_nickname_list = ['zheng07', 'leauthaud11', 'tinker13', 'hearin15']

    def __init__(self, model_nickname, **kwargs):
        """
        Parameters
        ----------
        model_nickname : string 
            String used to select the appropriate prebuilt 
            model_dictionary that will be used to build the instance. 
            See the ``Examples`` below. The list of available options are 

            * 'zheng07' (see :ref:`zheng07_composite_model` for a tutorial)
            
            * 'leauthaud11' (see :ref:`leauthaud11_composite_model`)

            * 'tinker13' (see :ref:`tinker13_composite_model`)

            * 'hearin15' (see :ref:`hearin15_composite_model`)

        halo_selection_func : function object, optional   
            Function object used to place a cut on the input ``table``. 
            If the ``halo_selection_func`` keyword argument is passed, 
            the input to the function must be a single positional argument storing a 
            length-N structured numpy array or Astropy table; 
            the function output must be a length-N boolean array that will be used as a mask. 
            Halos that are masked will be entirely neglected during mock population.

        Examples 
        ---------
        >>> from halotools.empirical_models import PrebuiltHodModelFactory
        >>> model_instance = PrebuiltHodModelFactory('zheng07')

        Passing in `zheng07` as the ``model_nickname`` argument triggers the factory to 
        call the `~halotools.empirical_models.zheng07_model_dictionary` function. 
        When doing so, the remaining arguments that were passed to the 
        `PrebuiltHodModelFactory` will in turn be passed on to 
        `~halotools.empirical_models.zheng07_model_dictionary`. 

        >>> model_instance = PrebuiltHodModelFactory('zheng07', threshold = -20)  

        The same applies to all pre-built models. 

        >>> model_instance = PrebuiltHodModelFactory('hearin15', threshold = 10.5, redshift = 2)

        Once you have built an instance of a composite model, you can use it to 
        populate any simulation in the Halotools cache: 

        >>> from halotools.sim_manager import CachedHaloCatalog # doctest: +SKIP
        >>> halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 2) # doctest: +SKIP
        >>> model_instance.populate_mock(halocat) # doctest: +SKIP

        As described in the `~halotools.empirical_models.ModelFactory.populate_mock` 
        docstring, calling the ``populate_mock`` method creates a ``mock`` attribute 
        bound to your model_instance. After you initially populate a halo catalog 
        using the `populate_mock` method, you can repopulate 
        the halo catalog by calling the 
        `~halotools.empirical_models.MockFactory.populate` method bound to 
        ``model_instance.mock``. 
        """
        input_model_dictionary, supplementary_kwargs = (
            self._retrieve_prebuilt_model_dictionary(model_nickname, **kwargs)
            )

        super_class_kwargs = {}
        for key, value in input_model_dictionary.items():
            super_class_kwargs[key] = value
        for key, value in supplementary_kwargs.items():
            super_class_kwargs[key] = value

        HodModelFactory.__init__(self, **super_class_kwargs)


    def _retrieve_prebuilt_model_dictionary(self, model_nickname, **constructor_kwargs):
        """
        """
        forbidden_constructor_kwargs = ('gal_type_list', 'model_feature_calling_sequence')
        for kwarg in forbidden_constructor_kwargs:
            if kwarg in constructor_kwargs:
                msg = ("\nWhen using the HodModelFactory to build an instance of a prebuilt model,\n"
                    "do not pass a ``%s`` keyword argument to the HodModelFactory constructor.\n"
                    "The appropriate source of this keyword is as part of a prebuilt model dictionary.\n")
                raise HalotoolsError(msg % kwarg)


        from ..composite_models import hod_models

        model_nickname = model_nickname.lower()

        if model_nickname == 'zheng07':
            dictionary_retriever = hod_models.zheng07_model_dictionary
        elif model_nickname == 'leauthaud11':
            dictionary_retriever = hod_models.leauthaud11_model_dictionary
        elif model_nickname == 'hearin15':
            dictionary_retriever = hod_models.hearin15_model_dictionary
        elif model_nickname == 'tinker13':
            dictionary_retriever = hod_models.tinker13_model_dictionary
        else:
            msg = ("\nThe ``%s`` model_nickname is not recognized by Halotools\n")
            raise HalotoolsError(msg % model_nickname)

        result = dictionary_retriever(**constructor_kwargs)
        if type(result) is dict:
            input_model_dictionary = result
            supplementary_kwargs = {}
            supplementary_kwargs['gal_type_list'] = None 
            supplementary_kwargs['model_feature_calling_sequence'] = None 
        elif type(result) is tuple:
            input_model_dictionary = result[0]
            supplementary_kwargs = result[1]
        else:
            raise HalotoolsError("Unexpected result returned from ``%s``\n"
            "Should be either a single dictionary or a 2-element tuple of dictionaries\n"
             % dictionary_retriever.__name__)

        return input_model_dictionary, supplementary_kwargs








