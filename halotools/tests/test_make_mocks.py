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
from ..halo_occupation import Satcen_Correlation_Polynomial_HOD_Model as satcen


def test_zheng_model():

	m = Zheng07_HOD_Model(threshold=-20.0)
	test_mass = np.array([10,11,12,13,14,15])
	test_mean_ncen = m.mean_ncen(test_mass)

	assert np.all(test_mean_ncen >= 0)

def test_satcen_mean_nsat():

	m = satcen(threshold=-19.5)
	# array of a few test masses
	p = np.arange(12,13,0.5)
	primary_halo_property = np.append(p,p)
	# array of halo_types
	h0 = np.zeros(len(p))
	h1 = np.zeros(len(p)) + 1
	halo_types = np.append(h0,h1)
	# arrays of indices for bookkeeping
	idx0=np.where(halo_types == 0)[0]
	idx1=np.where(halo_types == 1)[0]

	#probability of having halo_type=0
	phs0 = m.halotype_fraction_satellites(primary_halo_property[idx0],halo_types[idx0])
	#probability of having halo_type=1
	phs1 = m.halotype_fraction_satellites(primary_halo_property[idx1],halo_types[idx1])
	# Compute the value of mean_nsat that derives from the conditioned occupations
	derived_nsat = (phs0*m.mean_nsat(primary_halo_property[idx0],halo_types[idx0]) + 
		phs1*m.mean_nsat(primary_halo_property[idx1],halo_types[idx1]))
	# Compute the value of mean_nsat of the underlying baseline HOD model
	underlying_nsat = m.baseline_hod_model.mean_nsat(primary_halo_property[idx0])
	# Require that the derived and underlying 
	# satellite occupations are equal (highly non-trivial)
	assert np.all(derived_nsat == underlying_nsat)



