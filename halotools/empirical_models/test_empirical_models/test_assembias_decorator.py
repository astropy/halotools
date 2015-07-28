#!/usr/bin/env python

from unittest import TestCase
import pytest
from copy import copy

import numpy as np 
from astropy.table import Table

from ..abhod_components import HeavisideCenAssemBiasModel
from ...sim_manager import FakeSim
from ..hod_components import Zheng07Cens, Leauthaud11Cens
from .. import model_defaults
from ...utils.table_utils import SampleSelector


class TestAssembiasDecorator(TestCase):

    def setup_class(self):
    	pass

    def test_initialization(self):
    	pass


