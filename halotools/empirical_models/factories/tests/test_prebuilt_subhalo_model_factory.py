#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 

import numpy as np 
from copy import copy 

from ...smhm_models import Behroozi10SmHm, Moster13SmHm
from ...sfr_models import BinaryGalpropInterpolModel

from ...factories import PrebuiltSubhaloModelFactory, SubhaloModelFactory
from ...composite_models.smhm_models.behroozi10 import behroozi10_model_dictionary


from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ['TestPrebuiltSubhaloModelFactory']

class TestPrebuiltSubhaloModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.PrebuiltSubhaloModelFactory`. 
    """
    def test_behroozi_composite(self):
        """ Require that the `~halotools.empirical_models.composite_models.smhm_models.behroozi10_model_dictionary` 
        model dictionary builds without raising an exception. 
        """
        model = PrebuiltSubhaloModelFactory('behroozi10')
        alt_model = SubhaloModelFactory(**model.model_dictionary)

    def test_smhm_binary_sfr_composite(self):
        """ Require that the `~halotools.empirical_models.composite_models.sfr_models.smhm_binary_sfr_model_dictionary` 
        model dictionary builds without raising an exception. 
        """
        model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')
        alt_model = SubhaloModelFactory(**model.model_dictionary)
