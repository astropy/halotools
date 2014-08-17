#!/usr/bin/env python

"""
Very simple set of sanity checks on halo_occupation module 
Will copy and paste my additional tests once I figure out the basic design conventions.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)
import numpy as np
from ..halo_occupation import Zheng07_HOD_Model
from ..read_nbody import load_bolshoi_host_halos_fits
from ..make_mocks import HOD_mock


def test_Zheng07_mock():
	model = Zheng07_HOD_Model(threshold=-20)
	simulation = load_bolshoi_host_halos_fits()
	mock = HOD_mock(simulation,model)
	mock.populate()

	reasonable_ngal_boolean = (mock.num_total_gals > 5.e4) and (mock.num_total_gals < 1.e5)
	assert reasonable_ngal_boolean == True

	satellite_fraction = mock.num_total_sats/float(mock.num_total_gals)
	reasonable_satellite_fraction_boolean = (satellite_fraction > 0.1) and (satellite_fraction < 0.3)
	assert reasonable_satellite_fraction_boolean == True
