#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

import numpy as np 

from unittest import TestCase
import pytest 

from astropy.cosmology import WMAP9, Planck13, FLRW
from astropy import units as u

from ..profile_helpers import *

from ... import profile_models

from .....custom_exceptions import HalotoolsError


__all__ = ['TestAnalyticDensityProf']

class TestAnalyticDensityProf(TestCase):
    """
    """

    def setup_class(self):
        self.prof_model_list = (
            profile_models.NFWProfile, profile_models.TrivialProfile
            )

    def test_attrs(self):

        # Test that all sub-classes inherit the correct attributes
        for model_class in self.prof_model_list:
            model_instance = model_class(cosmology = WMAP9, redshift = 2, mdef = 'vir')


            assert hasattr(model_instance, 'cosmology')
            assert isinstance(model_instance.cosmology, FLRW)
            assert hasattr(model_instance, 'redshift')
            assert hasattr(model_instance, 'mdef')



