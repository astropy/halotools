#!/usr/bin/env python

import pytest
slow = pytest.mark.slow


import numpy as np
from astropy.config.paths import _find_home 

aph_home = u'/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

from .. import cache_config
from ..supported_sims import HaloCatalog 

@pytest.mark.skipif('not APH_MACHINE')
def test_load_halo_catalogs():

    simnames = cache_config.supported_sim_list
    zlist = [0, 0.5, 1, 2]
    for simname in simnames:
        for z in zlist:
            halocat = HaloCatalog(simname = simname, redshift = z)
