#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest

import numpy as np 
from copy import copy 

from ...factories import HodModelFactory

from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ['TestHodModelFactory']

class TestHodModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.SubhaloModelFactory`. 
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        pass

    def tearDown(self):
        pass
