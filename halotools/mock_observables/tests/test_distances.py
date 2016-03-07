#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..distances import periodic_3d_distance

__all__ = ['test_distances1']

def test_distances1():
	x1 = np.random.rand(5)
	y1 = np.random.rand(5)
	z1 = np.random.rand(5)
	x2 = np.random.rand(5)
	y2 = np.random.rand(5)
	z2 = np.random.rand(5)
	Lbox = 1
	d = periodic_3d_distance(x1, y1, z1, x2, y2, z2, Lbox)
