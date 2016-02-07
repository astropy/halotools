#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from copy import deepcopy 

from collections import Counter

import numpy as np 

from astropy.tests.helper import pytest
from astropy.table import Table 

from .. import catalog_analysis_helpers as cat_helpers

from ...sim_manager import FakeSim

from ...custom_exceptions import HalotoolsError

__all__ = ['TestCatalogAnalysisHelpers']

class TestCatalogAnalysisHelpers(TestCase):
    """ Class providing tests of the `~halotools.mock_observables.catalog_analysis_helpers`. 
    """
    
    def setUp(self):

        halocat = FakeSim()
        self.halo_table = halocat.halo_table

    def test_mean_y_vs_x1(self):
        abscissa, mean, err = cat_helpers.mean_y_vs_x(
            self.halo_table['halo_mvir'], self.halo_table['halo_spin'])

    def test_mean_y_vs_x2(self):
        abscissa, mean, err = cat_helpers.mean_y_vs_x(
            self.halo_table['halo_mvir'], self.halo_table['halo_spin'], 
            error_estimator = 'variance')

    def test_mean_y_vs_x3(self):
        with pytest.raises(HalotoolsError) as err:
            abscissa, mean, err = cat_helpers.mean_y_vs_x(
                self.halo_table['halo_mvir'], self.halo_table['halo_spin'], 
                error_estimator = 'Jose Canseco')
        substr = "Input ``error_estimator`` must be either"
        assert substr in err.value.message

    def tearDown(self):
        del self.halo_table









