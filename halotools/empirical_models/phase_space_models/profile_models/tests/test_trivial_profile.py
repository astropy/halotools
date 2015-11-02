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


__all__ = ['TestTrivialProfile']

class TestTrivialProfile(TestCase):
    """ Tests of `~halotools.empirical_models.halo_prof_components.TrivialProfile`. 

    Mostly this function checks that the each of the following attributes is present, 
    and is an empty array, list, or dictionary:

        * ``cumu_inv_func_table``

        * ``cumu_inv_func_table_dict``

        * ``cumu_inv_param_table``

    """
    def setup_class(self):
        pass

    # The following tests are useful but need to be rewritten according to 
    # the changes made by the prof_overhaul branch 

    # # Check that the initialized attributes are correct
    # model_instance = hpc.TrivialProfile()
    # assert model_instance.prof_param_keys == []
    
    # # Check that the lookup table attributes are correct
    # model_instance.build_inv_cumu_lookup_table()
    # assert len(model_instance.cumu_inv_func_table) == 0
    # assert len(model_instance.func_table_indices) == 0

