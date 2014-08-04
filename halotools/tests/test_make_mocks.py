#!/usr/bin/env python

"""
Very simple set of sanity checks on mock.py. 
Still figuring out how to structure this properly.
Will copy and paste my additional tests once I figure out the basic design conventions.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
from ..halo_occupation import Zheng07_HOD_Model


def test_zheng_model():

	m = Zheng07_HOD_Model(threshold=-20.0)
	test_mass = np.array([10,11,12,13,14,15])
	test_mean_ncen = m.mean_ncen(test_mass)

	assert np.all(test_mean_ncen >= 0)




