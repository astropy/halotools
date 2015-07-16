#!/usr/bin/env python

from ..read_nbody import ProcessedSnapshot
import os
from astropy.config.paths import _find_home 

def test_cached_simulations():
	xch = os.environ.get('XDG_CACHE_HOME')
	home = _find_home()
	astropy_cache = os.path.join(home, '.astropy')
	os.environ['XDG_CACHE_HOME'] = astropy_cache
	simobj = ProcessedSnapshot()
	os.environ['XDG_CACHE_HOME'] = xch
	