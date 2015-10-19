#!/usr/bin/env python

import pytest
from unittest import TestCase
import numpy as np 
from astropy.table import Table 

from ..phase_space_models import NFWPhaseSpace

from ....sim_manager import HaloCatalog
from ....custom_exceptions import HalotoolsError
from ..velocity_models import NFWJeansVelocity

class TestNFWJeansVelocity(TestCase):
    """ Class used to test `~halotools.empirical_models.NFWPhaseSpace`. 
    """

    def setup_class(self):
        """ Load the NFW model and build a coarse lookup table.
        """
        pass

    def test_velocity_dispersion(self):
        """
        """
        pass
