#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

from unittest import TestCase
import pytest 

import numpy as np 
from copy import copy 

from ...factories import SubhaloModelFactory
from ...composite_models.smhm_models.behroozi10 import behroozi10_model_dictionary

from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ['TestSubhaloModelFactory']

class TestSubhaloModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.factories.SubhaloModelFactory`. 
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """

    def test_behroozi_composite(self):
        """ Require that the `~halotools.empirical_models.composite_models.smhm_models.behroozi10_model_dictionary` 
        model dictionary builds without raising an exception. 
        """
        model = SubhaloModelFactory('behroozi10')
        alt_model = SubhaloModelFactory(**model.model_dictionary)

    def test_exception_handling(self):
        """ Using `~halotools.empirical_models.composite_models.smhm_models.behroozi10_model_dictionary` 
        as a specific example, require that the `~halotools.empirical_models.SubhaloModelFactory` raises a HalotoolsError 
        if any of its component models do not have `_galprop_dtypes_to_allocate` and `_methods_to_inherit` attributes. 
        """

        model_dictionary = behroozi10_model_dictionary()

        model = SubhaloModelFactory(**model_dictionary)

        tmp = copy(model_dictionary['stellar_mass']._galprop_dtypes_to_allocate)
        del model_dictionary['stellar_mass']._galprop_dtypes_to_allocate
        with pytest.raises(HalotoolsError):
            model = SubhaloModelFactory(**model_dictionary)
        model_dictionary['stellar_mass']._galprop_dtypes_to_allocate = tmp

        tmp = copy(model_dictionary['stellar_mass']._methods_to_inherit)
        del model_dictionary['stellar_mass']._methods_to_inherit
        with pytest.raises(HalotoolsError):
            model = SubhaloModelFactory(**model_dictionary)

    def test_smhm_binary_sfr_composite(self):
        """ Require that the `~halotools.empirical_models.composite_models.sfr_models.smhm_binary_sfr_model_dictionary` 
        model dictionary builds without raising an exception. 
        """
        model = SubhaloModelFactory('smhm_binary_sfr')
        alt_model = SubhaloModelFactory(**model.model_dictionary)


