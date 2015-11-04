#!/usr/bin/env python

import pytest
from unittest import TestCase
import numpy as np 
from astropy.table import Table 

from ..nfw_isotropic_jeans import NFWJeansVelocity

from ...nfw_phase_space import NFWPhaseSpace

from .....sim_manager import HaloCatalog
from .....custom_exceptions import HalotoolsError

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
