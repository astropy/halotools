#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

import numpy as np 

from unittest import TestCase
import pytest 

from astropy.cosmology import WMAP9, Planck13
from astropy import units as u

from ..trivial_profile import TrivialProfile

from .... import model_defaults 

from .....custom_exceptions import HalotoolsError
from .....sim_manager import sim_defaults


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
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.default_model = TrivialProfile()
        self.wmap9_model = TrivialProfile(cosmology = WMAP9)
        self.m200_model = TrivialProfile(mdef = '200m')

    def test_instance_attrs(self):
        """ Require that all model variants have ``cosmology``, ``redshift`` and ``mdef`` attributes. 
        """
        assert self.default_model.cosmology == sim_defaults.default_cosmology
        assert self.m200_model.cosmology == sim_defaults.default_cosmology
        assert self.wmap9_model.cosmology == WMAP9

        assert self.default_model.redshift == sim_defaults.default_redshift
        assert self.m200_model.redshift == sim_defaults.default_redshift
        assert self.wmap9_model.redshift == sim_defaults.default_redshift

        assert self.default_model.mdef == model_defaults.halo_mass_definition
        assert self.m200_model.mdef == '200m'
        assert self.wmap9_model.mdef == model_defaults.halo_mass_definition


