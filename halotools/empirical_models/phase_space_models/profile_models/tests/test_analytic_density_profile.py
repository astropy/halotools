#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

import numpy as np 

from unittest import TestCase
import pytest 

from astropy.cosmology import WMAP9, Planck13
from astropy import units as u

from ..profile_helpers import *
from .....custom_exceptions import HalotoolsError


__all__ = ['TestAnalyticDensityProf']

class TestAnalyticDensityProf(TestCase):

    def setup_class(self):
        pass

    # The following tests are useful but need to be rewritten according to 
    # the changes made by the prof_overhaul branch 

    # prof_model_list = hpc.__all__
    # parent_class = hpc.HaloProfileModel

    # # First create a list of all sub-classes to test
    # component_models_to_test = []
    # for clname in prof_model_list:
    #     cl = getattr(hpc, clname)

    #     if (issubclass(cl, parent_class)) & (cl != parent_class):
    #         component_models_to_test.append(cl)

    # # Now we will test that all sub-classes inherit the correct behavior
    # for model_class in component_models_to_test:
    #     model_instance = model_class(cosmology=cosmology.WMAP7, redshift=2)

    #     assert hasattr(model_instance, 'cosmology')
    #     assert isinstance(model_instance.cosmology, cosmology.FlatLambdaCDM)
    #     assert hasattr(model_instance, 'redshift')

    #     assert hasattr(model_instance, 'build_inv_cumu_lookup_table')
    #     model_instance.build_inv_cumu_lookup_table()
    #     assert hasattr(model_instance, 'cumu_inv_func_table')
    #     assert type(model_instance.cumu_inv_func_table) == np.ndarray
    #     assert hasattr(model_instance, 'func_table_indices')
    #     assert type(model_instance.func_table_indices) == np.ndarray




