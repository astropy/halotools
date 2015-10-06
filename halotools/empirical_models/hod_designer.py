# -*- coding: utf-8 -*-
"""
This module provides a convenient interface 
for building a new galaxy-halo model by swapping out features 
from an existing model. 
"""

from copy import copy 
from warnings import warn 

from .model_factories import HodModelFactory 

from ..custom_exceptions import *

__all__ = ['HodModelArchitect']


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

        for new_component in args:
            try:
                gal_type = new_component.gal_type
                galprop_key = new_component.galprop_key
            except AttributeError:
                msg = ("\nEvery argument of the customize_model method of HodModelArchitect "
                    "must be a model instance that has a ``gal_type`` and a ``galprop_key`` attribute.\n")
                raise HalotoolsError(msg)

            # Enforce self-consistency in the thresholds of new and old components
            if galprop_key == 'occupation':
                old_component = baseline_blueprint[gal_type][galprop_key]
                if new_component.threshold != old_component.threshold:
                    msg = ("\n\nYou tried to swap in a %s occupation component \nthat has a different " 
                        "threshold than the original %s occupation component.\n"
                        "This is technically permissible, but in general, composite HOD-style models \n"
                        "must have the same threshold for all occupation components.\n"
                        "Thus if you do not request the HodModelArchitect to make the corresponding threshold change \n"
                        "for all gal_types, the resulting composite model will raise an exception and not build.\n")
                    warn(msg % (gal_type, gal_type)) 

            new_blueprint[gal_type][galprop_key] = new_component

        new_model = HodModelFactory(new_blueprint)
        return new_model








