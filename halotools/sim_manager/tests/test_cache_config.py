#!/usr/bin/env python
import numpy as np
import os
import unittest
from astropy.tests.helper import pytest

from .. import cache_config

__all__ = (
	['test_cache_config', 'test_catalogs_config', 'test_should_not_create_dir', 
	'test_supported_simnames', 'test_supported_halo_finders']
	)

def test_cache_config():
	""" Verify that the Astropy and Halotools 
	cache directories are detected, and that the latter 
	is a subdirectory of the former. 
	"""
	astropy_cache = cache_config.get_astropy_cache_dir()
	assert os.path.exists(astropy_cache)
	halotools_cache = cache_config.get_halotools_cache_dir()
	assert os.path.exists(halotools_cache)
	assert os.path.join(astropy_cache, 'halotools') == halotools_cache

def test_catalogs_config():
	""" Verify that the raw_halo_catalogs and halo_catalogs Halotools 
	cache directories are detected, and that they are subdirectories of  
	Halotools cache. 
	"""
	halotools_cache = cache_config.get_halotools_cache_dir()

	raw_halos_subdir = cache_config.get_catalogs_dir('raw_halos')
	assert os.path.exists(raw_halos_subdir)
	assert os.path.join(halotools_cache, 'raw_halo_catalogs') == raw_halos_subdir

	halos_subdir = cache_config.get_catalogs_dir('halos')
	assert os.path.exists(halos_subdir)
	assert os.path.join(halotools_cache, 'halo_catalogs') == halos_subdir


def test_should_not_create_dir():
	""" Require that attempting to create a cache subdirectory 
	for an unsupported simulation and/or halo-finder raises an IOError. 
	"""
	with pytest.raises(IOError) as exc:
		parent_dir = cache_config.get_catalogs_dir('raw_halos')
		nonsense_dirname = 'JoseCanseco'
		func = cache_config.cache_subdir_for_simulation
		s1 = func(parent_dir, nonsense_dirname)

	exception_string = ("It is not permissible to create a subdirectory of " + 
		"Halotools cache \nfor simulations which have no class defined in " + 
		"the halotools/sim_manager/supported_sims module. \n")
	assert exc.value.args[0] == exception_string

	with pytest.raises(IOError) as exc:
		parent_dir = cache_config.get_catalogs_dir('raw_halos')
		nonsense_dirname = 'JoseCanseco'
		func = cache_config.cache_subdir_for_halo_finder
		s1 = func(parent_dir, 'bolshoi', nonsense_dirname)
	exception_string = ("It is not permissible to create a subdirectory of "
                "Halotools cache \nfor a combination of "
                "simulation + halo-finder which has no corresponding class defined in "
                "the halotools/sim_manager/supported_sims module. \n")
	assert exc.value.args[0] == exception_string


def test_supported_simnames():
	""" Require `bolshoi`, `bolshoipl`, and `multidark` to 
	appear in the list of supported simulations. 
	"""
	hflist = cache_config.get_supported_simnames()
	assert 'bolshoi' in hflist
	assert 'bolshoiplanck' in hflist
	assert 'multidark' in hflist

def test_supported_halo_finders():
	""" Require `rockstar` to 
	appear in the list of supported halo-finders for all simulations. 
	"""
	simlist = cache_config.get_supported_simnames()
	for sim in simlist:
		hflist = cache_config.get_supported_halo_finders(sim)
		assert 'rockstar' in hflist
		if sim == 'bolshoi':
			assert 'bdm' in hflist








