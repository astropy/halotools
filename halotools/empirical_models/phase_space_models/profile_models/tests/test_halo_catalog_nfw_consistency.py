#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import numpy as np

from unittest import TestCase
from astropy.tests.helper import pytest

from ..nfw_profile import NFWProfile

from .... import model_defaults

from .....sim_manager import CachedHaloCatalog
from .....utils import table_utils


### Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
from astropy.config.paths import _find_home
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


__all__ = ['TestHaloCatNFWConsistency']


class TestHaloCatNFWConsistency(TestCase):
    """ Tests of `~halotools.empirical_models.NFWProfile` in which comparisons are made to a Bolshoi halo catalog.

    """
    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests.
        """
        halocat = CachedHaloCatalog(simname='bolshoi', redshift=0.)
        hosts = table_utils.SampleSelector.host_halo_selection(table=halocat.halo_table)

        mask_mvir_1e11 = (hosts['halo_mvir'] > 1e11) & (hosts['halo_mvir'] < 2e11)
        self.halos_mvir_1e11 = hosts[mask_mvir_1e11]

        mask_mvir_1e12 = (hosts['halo_mvir'] > 1e12) & (hosts['halo_mvir'] < 2e12)
        self.halos_mvir_1e12 = hosts[mask_mvir_1e12]

        mask_mvir_1e13 = (hosts['halo_mvir'] > 1e13) & (hosts['halo_mvir'] < 2e13)
        self.halos_mvir_1e13 = hosts[mask_mvir_1e13]

        self.halo_sample_names = ['halos_mvir_1e11', 'halos_mvir_1e12', 'halos_mvir_1e13']

        self.nfw_profile = NFWProfile(cosmology=halocat.cosmology,
            redshift=0., mdef='vir')

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_conc_nfw_consistency(self):

        for sample_name in self.halo_sample_names:
            halos = getattr(self, sample_name)

            cmin = model_defaults.min_permitted_conc
            cmax = model_defaults.max_permitted_conc
            carr = halos[model_defaults.concentration_key]
            mask = (carr >= cmin) & (carr <= cmax)

            median_conc = np.median(carr[mask])

            predicted_conc = self.nfw_profile.conc_NFWmodel(
                table=halos[mask])

            assert np.allclose(carr[mask], predicted_conc, rtol=0.001)

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_vmax_consistency(self):

        for sample_name in self.halo_sample_names:
            halos = getattr(self, sample_name)

            median_vmax = np.median(halos['halo_vmax'])
            median_mass = np.median(halos['halo_mvir'])
            median_conc = np.median(halos['halo_nfw_conc'])

            predicted_vmax = self.nfw_profile.vmax(
                median_mass, median_conc)

            assert np.allclose(median_vmax, predicted_vmax, rtol=0.05)
