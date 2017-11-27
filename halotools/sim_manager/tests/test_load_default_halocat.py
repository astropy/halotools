"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals


from astropy.config.paths import _find_home
import pytest

from ..cached_halo_catalog import CachedHaloCatalog
from ..ptcl_table_cache import PtclTableCache
from ..halo_table_cache import HaloTableCache

from ...utils.python_string_comparisons import compare_strings_py23_safe

aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False
__all__ = ('test_load_default_halocat', )


@pytest.mark.skipif('not APH_MACHINE')
def test_load_default_halocat():
    """
    """
    halocat = CachedHaloCatalog()
    particles = halocat.ptcl_table
    halos = halocat.halo_table


@pytest.mark.skipif('not APH_MACHINE')
def test_load_all_cached_halocats_from_fname():
    """
    """
    cache = HaloTableCache()
    for entry in cache.log:
        fname = entry.fname
        halocat = CachedHaloCatalog(fname=fname)
