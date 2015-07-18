#!/usr/bin/env python

from ..read_nbody import ProcessedSnapshot
from ..cache_config import enable_cache_access_during_pytest

"""
# This code seems to delete the entire cache! 

@enable_cache_access_during_pytest
def test_cached_simulations():
	simobj = ProcessedSnapshot()
"""