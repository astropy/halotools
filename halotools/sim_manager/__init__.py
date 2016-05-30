# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" The `~halotools.sim_manager` sub-package is responsible
for downloading halo catalogs, reading ascii data,
storing hdf5 binaries and keeping a persistent memory
of their location on disk and associated metadata.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from astropy.config.paths import _find_home

try:
    halotools_cache_dirname = os.path.join(_find_home(), '.astropy', 'cache', 'halotools')
    os.makedirs(halotools_cache_dirname)
except OSError:
    pass

from .fake_sim import FakeSim

from .download_manager import DownloadManager

from .cached_halo_catalog import CachedHaloCatalog
from .user_supplied_halo_catalog import UserSuppliedHaloCatalog
from .user_supplied_ptcl_catalog import UserSuppliedPtclCatalog

from .rockstar_hlist_reader import RockstarHlistReader
from .tabular_ascii_reader import TabularAsciiReader
from .halo_table_cache import HaloTableCache
from .ptcl_table_cache import PtclTableCache
