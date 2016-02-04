#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import warnings, os, shutil

from astropy.config.paths import _find_home 
from astropy.tests.helper import remote_data, pytest


import numpy as np 

from ..fake_sim import FakeSim, FakeSimHalosNearBoundaries
from ...custom_exceptions import HalotoolsError

__all__ = ['TestFakeSim', 'TestFakeSimHalosNearBoundaries']

class TestFakeSim(TestCase):
	"""
	"""
	def setUp(self):
		self.fake_sim = FakeSim()

	def test_attrs(self):
		keylist = ['halo_hostid']
		for key in keylist:
			assert key in self.fake_sim.halo_table.keys()

class TestFakeSimHalosNearBoundaries(TestCase):
	"""
	"""
	def setUp(self):
		self.fake_sim = FakeSimHalosNearBoundaries()

	def test_attrs(self):
		keylist = ['halo_hostid']
		for key in keylist:
			assert key in self.fake_sim.halo_table.keys()

	def test_positions(self):
		assert not np.any( (self.fake_sim.halo_table['halo_x'] > 1) & 
			(self.fake_sim.halo_table['halo_x'] < self.fake_sim.Lbox - 1) )




