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

APH_MACHINE = False

@pytest.mark.skipif('not APH_MACHINE')
def test_load_halo_catalogs():

    simnames = cache_config.supported_sim_list
    adict = {'bolshoi': [0.33035, 0.54435, 0.67035, 1], 'bolplanck': [0.33406, 0.50112, 0.67, 1], 
        'consuelo': [0.333, 0.506, 0.6754, 1], 'multidark': [0.318, 0.5, 0.68, 1]}
    for simname in simnames:
        alist = adict[simname]
        for a in alist:
            z = 1/a - 1
            halocat = HaloCatalog(simname = simname, redshift = z)
            halos = halocat.halo_table
            if simname not in ['bolshoi', 'multidark']:
                particles = halocat.ptcl_table
            else:
                if a == 1:
                    particles = halocat.ptcl_table

