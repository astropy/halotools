#!/usr/bin/env python

import numpy as np 
from astropy.table import Table 
from astropy.io.ascii import read as astropy_ascii_read
from astropy.utils.data import get_pkg_data_filename, get_pkg_data_fileobj
from astropy.tests.helper import remote_data, pytest
from unittest import TestCase
from copy import copy


from ...smhm_models import *
from ... import model_defaults

from ....sim_manager import sim_defaults
from ....custom_exceptions import *

def test_behroozi10_redshift_safety():
	"""
	"""
	model = Behroozi10SmHm()

	result0 = model.mean_log_halo_mass(11)
	result1 = model.mean_log_halo_mass(11, redshift = 4)
	result2 = model.mean_log_halo_mass(11, redshift = sim_defaults.default_redshift)
	assert result0 == result2
	assert result0 != result1

	result0 = model.mean_stellar_mass(prim_haloprop = 1e12)
	result1 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = 4)
	result2 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = sim_defaults.default_redshift)
	assert result0 == result2
	assert result0 != result1

	model = Behroozi10SmHm(redshift = sim_defaults.default_redshift)
	result0 = model.mean_log_halo_mass(11)
	with pytest.raises(HalotoolsError) as exc:
		result1 = model.mean_log_halo_mass(11, redshift = 4)
	result2 = model.mean_log_halo_mass(11, redshift = model.redshift)
	assert result0 == result2

	result0 = model.mean_stellar_mass(prim_haloprop = 1e12)
	with pytest.raises(HalotoolsError) as exc:
		result1 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = 4)
	result2 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = model.redshift)
	assert result0 == result2




