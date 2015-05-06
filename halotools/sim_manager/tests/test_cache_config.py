#!/usr/bin/env python
import numpy as np
import os

from .. import cache_config

__all__ = ['test_cache_config']

def test_cache_config():
	astropy_cache = cache_config.get_astropy_cache_dir()
	halotools_cache = cache_config.get_halotools_cache_dir()
	assert os.path.join(astropy_cache, 'halotools') == halotools_cache

def test_catalogs_config():
	halotools_cache = cache_config.get_halotools_cache_dir()
		


