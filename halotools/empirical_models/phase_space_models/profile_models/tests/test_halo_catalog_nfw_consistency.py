#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

import numpy as np 

from unittest import TestCase
import pytest 

from astropy.cosmology import WMAP9, Planck13
from astropy import units as u

from .. import profile_helpers

from .....sim_manager import HaloCatalog
from .....utils import table_utils

from .....custom_exceptions import HalotoolsError


### Determine whether the machine is mine
# This will be used to select tests whose 
# returned values depend on the configuration 
# of my personal cache directory files
from astropy.config.paths import _find_home 
aph_home = u'/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


__all__ = ['TestHaloCatalogNFWConsistency']

class TestHaloCatalogNFWConsistency(TestCase):
    """ Tests of `~halotools.empirical_models.halo_prof_components.NFWProfile` 
    in which comparisons are made to a halo catalog. 

    """
    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def setup_class(self):
        """
        """
        halocat = HaloCatalog(simname = 'bolshoi', redshift = 0)
        hosts = table_utils.SampleSelector.host_halo_selection(
            table = halocat.halo_table)

        mask_mvir_1e11 = (hosts['halo_mvir'] > 1e11) & (hosts['halo_mvir'] < 2e11)
        self.halos_mvir_1e11 = hosts[mask_mvir_1e11]

        mask_mvir_1e12 = (hosts['halo_mvir'] > 1e12) & (hosts['halo_mvir'] < 2e12)
        self.halos_mvir_1e12 = hosts[mask_mvir_1e12]

        mask_mvir_1e13 = (hosts['halo_mvir'] > 1e13) & (hosts['halo_mvir'] < 2e13)
        self.halos_mvir_1e13 = hosts[mask_mvir_1e13]


    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_vmax_consistency(self):
        pass
        













