#!/usr/bin/env python

from unittest import TestCase
import numpy as np

from astropy.config.paths import _find_home
from astropy.tests.helper import pytest

from ..cached_halo_catalog import CachedHaloCatalog

slow = pytest.mark.slow

aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ('TestSupportedSims', )


class TestSupportedSims(TestCase):
    """ Class providing unit testing for `~halotools.sim_manager.HaloTableCacheLogEntry`.
    """
    adict = {'bolshoi': [0.33035, 0.54435, 0.67035, 1], 'bolplanck': [0.33406, 0.50112, 0.67, 1],
        'consuelo': [0.333, 0.506, 0.6754, 1], 'multidark': [0.318, 0.5, 0.68, 1]}

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_halo_catalogs(self):
        """
        """

        for simname in list(self.adict.keys()):
            alist = self.adict[simname]
            for a in alist:
                z = 1/a - 1
                halocat = CachedHaloCatalog(simname=simname, redshift=z)
                assert np.allclose(halocat.redshift, z, atol=0.01)

                if simname not in ['bolshoi', 'multidark']:
                    particles = halocat.ptcl_table
                else:
                    if a == 1:
                        particles = halocat.ptcl_table

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_halo_rvir_in_correct_units(self):
        """ Loop over all halo catalogs in cache and verify that the
        ``halo_rvir`` column never exeeds the number 50. This is a crude way of
        ensuring that units are in Mpc/h, not kpc/h.
        """
        for simname in list(self.adict.keys()):
            alist = self.adict[simname]
            a = alist[0]
            z = 1/a - 1
            halocat = CachedHaloCatalog(simname=simname, redshift=z)
            r = halocat.halo_table['halo_rvir']
            assert np.all(r < 50.)
