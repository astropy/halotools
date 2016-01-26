#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest

import numpy as np 
from copy import copy 

from ...factories import PrebuiltHodModelFactory


from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ['TestPrebuiltHodModelFactory']

class TestPrebuiltHodModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.PrebuiltHodModelFactory`. 
    """
    @pytest.mark.slow
    def test_fake_mock_population(self):
        for modelname in PrebuiltHodModelFactory.prebuilt_model_nickname_list:
            model = PrebuiltHodModelFactory(modelname)
            model.populate_mock(simname = 'fake')

