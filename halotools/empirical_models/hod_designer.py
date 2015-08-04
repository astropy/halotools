# -*- coding: utf-8 -*-
"""

This module provides a convenient interface 
for building a new galaxy-halo model by swapping out features 
from an existing model. 
"""

from ..halotools_exceptions import HalotoolsError

class HodModelArchitect(object):

    def __init__(self):
        pass

    @staticmethod
    def customize_model(*args, **kwargs):

        try:
            baseline_model = kwargs['baseline_model']
        except KeyError:
            msg = ("\nThe customize_model method of HodModelArchitect "
                "requires a baseline_model keyword argument\n")
            raise HalotoolsError(msg)
        baseline_blueprint = baseline_model.model_blueprint

        for component in args:
            try:
                gal_type = component.gal_type
            except AttributeError:
                msg = ("\nEvery argument of the customize_model method of HodModelArchitect "
                    "must be a model instance that has a ``gal_type`` attribute.\n")
                raise HalotoolsError(msg)







