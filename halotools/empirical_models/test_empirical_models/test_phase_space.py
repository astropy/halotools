#!/usr/bin/env python

from unittest import TestCase

import numpy as np 
from astropy.table import Table 
from ...sim_manager import HaloCatalog
from ..phase_space_models import NFWPhaseSpace

class TestAssembias(TestCase):
    """
    """

    def setup_class(self):
        """
        """
        self.halocat = HaloCatalog()

        self.nfw = NFWPhaseSpace()
        cmin, cmax, dc = 1, 15, 15
        self.nfw._setup_lookup_tables((cmin, cmax, dc))

    def test_nfw_phase_space(self):
        pass
        










