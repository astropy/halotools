#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

from unittest import TestCase
import pytest 

import numpy as np 
from copy import copy 

from ...factories import SubhaloModelFactory

from ....sim_manager import FakeSim

class TestSubhaloModelFactory(TestCase):

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """

    def test_behroozi_composite(self):
        model = SubhaloModelFactory('behroozi10')
        alt_model = SubhaloModelFactory(**model.model_dictionary)

    def test_smhm_binary_sfr_composite(self):
        model = SubhaloModelFactory('smhm_binary_sfr')
        alt_model = SubhaloModelFactory(**model.model_dictionary)


