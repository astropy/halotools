#!/usr/bin/env python
import numpy as np
import os
import unittest
from astropy.tests.helper import pytest

from .. import cache_config
from ...custom_exceptions import UnsupportedSimError

from astropy.config.paths import _find_home

__all__ = (
	['test_cache_config', 'test_catalogs_config', 
	'test_supported_simnames', 'test_supported_halo_finders']
	)

def test_cache_config():
	""" Verify that the Astropy and Halotools 
	cache directories are detected, and that the latter 
	is a subdirectory of the former. 
	"""

	homedir = _find_home()
	astropy_cache_dir = os.path.join(homedir, '.astropy', 'cache')
	if not os.path.isdir(astropy_cache_dir):
		os.mkdir(astropy_cache_dir)
		
	halotools_cache = cache_config.get_catalogs_dir()
	assert os.path.exists(halotools_cache)
	assert os.path.join(astropy_cache_dir, 'halotools') == halotools_cache

def test_catalogs_config():
	""" Verify that the raw_halo_catalogs and halo_catalogs Halotools 
	cache directories are detected, and that they are subdirectories of  
	Halotools cache. 
	"""
	halotools_cache = cache_config.get_catalogs_dir()

	raw_halos_subdir = cache_config.get_catalogs_dir(catalog_type='raw_halos')
	assert os.path.exists(raw_halos_subdir)
	assert os.path.join(halotools_cache, 'raw_halo_catalogs') == raw_halos_subdir

	halos_subdir = cache_config.get_catalogs_dir(catalog_type='halos')
	assert os.path.exists(halos_subdir)
	assert os.path.join(halotools_cache, 'halo_catalogs') == halos_subdir


def test_supported_simnames():
	""" Require `bolshoi`, `bolshoipl`, and `multidark` to 
	appear in the list of supported simulations. 
	"""
	hflist = cache_config.supported_sim_list
	assert 'bolshoi' in hflist
	assert 'bolplanck' in hflist
	assert 'multidark' in hflist
	assert 'consuelo' in hflist

def test_supported_halo_finders():
	""" Require `rockstar` to 
	appear in the list of supported halo-finders for all simulations. 
	"""
	simlist = cache_config.supported_sim_list
	for sim in simlist:
		hflist = cache_config.get_supported_halo_finders(sim)
		assert 'rockstar' in hflist
		if sim == 'bolshoi':
			assert 'bdm' in hflist

	with pytest.raises(UnsupportedSimError) as exc:
		x = cache_config.get_supported_halo_finders('JoseCanseco')








