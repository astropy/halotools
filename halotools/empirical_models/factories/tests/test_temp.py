#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest
from astropy.config.paths import _find_home 

import numpy as np 
from copy import copy 

from ....sim_manager import FakeSim
from ....sim_manager.fake_sim import FakeSimHalosNearBoundaries
from ..prebuilt_model_factory import PrebuiltHodModelFactory
from ....custom_exceptions import HalotoolsError

def test_zero_satellite_edge_case():
    model = PrebuiltHodModelFactory('zheng07', threshold = -18)
    model.param_dict['logM0'] = 20

    halocat = FakeSim()
    model.populate_mock(halocat = halocat)

