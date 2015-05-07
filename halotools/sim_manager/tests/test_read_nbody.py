#!/usr/bin/env python
import numpy as np
from astropy.tests.helper import pytest

from .. import read_nbody

__all__ = ['test_catalog_manager', 'TestDummyClass']

def test_catalog_manager():
	catman = read_nbody.CatalogManager()


class TestDummyClass(object):

	@pytest.mark.marf
	def test_dummy(self):
		assert 5==5

