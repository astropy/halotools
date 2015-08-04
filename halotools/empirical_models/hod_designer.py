# -*- coding: utf-8 -*-
"""
This module provides a convenient interface 
for building a new galaxy-halo model by swapping out features 
from an existing model. 
"""

from copy import copy 
from ..halotools_exceptions import HalotoolsError
from .model_factories import HodModelFactory 


class HodModelArchitect(object):
    """ Class used to create customized HOD-style models.  
    """

    def __init__(self):
        pass

    @staticmethod
    def customize_model(*args, **kwargs):
        """ Method takes a baseline composite model as input, 
        together with an arbitrary number of new component models, 
        and swaps in the new component models to create a and return new composite model. 

        Parameters 
        ----------
        baseline_model : HOD model instance 
            `~halotools.empirical_models.HodModelFactory` instance. 

        component_models : Halotools objects 
            Instance of any component model that you want to swap in to the baseline_model. 

        Returns 
        --------
        new_model : HOD model instance  
            `~halotools.empirical_models.HodModelFactory` instance. The ``new_model`` will 
            be identical in every way to the ``baseline_model``, except the features in the 
            input component_models will replace the features in the ``baseline_model``. 

        """

        try:
            baseline_model = kwargs['baseline_model']
        except KeyError:
            msg = ("\nThe customize_model method of HodModelArchitect "
                "requires a baseline_model keyword argument\n")
            raise HalotoolsError(msg)
        baseline_blueprint = baseline_model.model_blueprint
        new_blueprint = copy(baseline_blueprint)

        for component in args:
            try:
                gal_type = component.gal_type
                galprop_key = component.galprop_key
            except AttributeError:
                msg = ("\nEvery argument of the customize_model method of HodModelArchitect "
                    "must be a model instance that has a ``gal_type`` and a ``galprop_key`` attribute.\n")
                raise HalotoolsError(msg)
            new_blueprint[gal_type][galprop_key] = component
        new_model = HodModelFactory(new_blueprint)

        return new_model








