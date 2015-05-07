#!/usr/bin/env python

import numpy as np
import os

from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_fileobj

from .. import read_nbody, cache_config


__all__ = ['test_closest_halocat_in_cache', 'test_available_snapshots_in_cache', 'TestDummyClass']



def test_closest_halocat_in_cache():
	catman = read_nbody.CatalogManager()
   	f = catman.closest_halocat_in_cache(
   		'halos', 'bolshoi', 'rockstar', 100)
   	#assert f != None
   	#assert f[0] == u'/Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_0.09630.list.mpeak.gt.2e9.hdf5'

def test_available_snapshots_in_cache():
	catman = read_nbody.CatalogManager()
   	flist = catman.available_snapshots('cache', 'halos', 'bolshoi', 'rockstar')
   	#assert len(flist) != 0
   	#f = os.path.basename(flist[0])
   	correct_fname = u'hlist_0.09630.list.mpeak.gt.2e9.hdf5'



class TestDummyClass(object):

	@pytest.mark.marf
	def test_dummy(self):
		assert 5==5




		#f = get_pkg_data_fileobj()




