#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 

import numpy as np 
from copy import copy 

from .. import UserDefinedHaloCatalog

__all__ = ['TestUserDefinedHaloCatalog']

class TestUserDefinedHaloCatalog(TestCase):
    """ Class providing tests of the `~halotools.sim_manager.UserDefinedHaloCatalog`. 
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        pass

    def test_metadata(self):
        pass
