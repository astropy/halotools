"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.config.paths import _find_home
import pytest

from ..cached_halo_catalog import CachedHaloCatalog, InvalidCacheLogEntry

from ...custom_exceptions import HalotoolsError

slow = pytest.mark.slow

aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

__all__ = ('test_load_halo_catalogs', 'test_halo_rvir_in_correct_units')

adict = {'bolshoi': [0.33035, 0.54435, 0.67035, 1], 'bolplanck': [0.33406, 0.50112, 0.67, 1],
    'consuelo': [0.333, 0.506, 0.6754, 1], 'multidark': [0.318, 0.5, 0.68, 1]}


@pytest.mark.slow
@pytest.mark.skipif('not APH_MACHINE')
def test_load_halo_catalogs():
    """
    """

    for simname in list(adict.keys()):
        alist = adict[simname]
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
def test_halo_rvir_in_correct_units():
    """ Loop over all halo catalogs in cache and verify that the
    ``halo_rvir`` column never exeeds the number 50. This is a crude way of
    ensuring that units are in Mpc/h, not kpc/h.
    """
    for simname in list(adict.keys()):
        alist = adict[simname]
        a = alist[0]
        z = 1/a - 1
        halocat = CachedHaloCatalog(simname=simname, redshift=z)
        r = halocat.halo_table['halo_rvir']
        assert np.all(r < 50.)


def test_bolplanck_particle_mass():
    """ This is a regression test for https://github.com/astropy/halotools/issues/576

    This test should never be deleted or refactored.
    """
    from ..supported_sims import BolPlanck
    bp = BolPlanck()
    assert np.allclose(bp.particle_mass, 1.55e8, rtol=0.01)


@pytest.mark.skipif('not HAS_H5PY')
def test_forbidden_sims():
    """ This is a regression test that ensures no one will use the
    z = 0 halotools_alpha_version2 halo catalog.

    This test should never be deleted or refactored.
    """
    with pytest.raises(HalotoolsError) as err:
        __ = CachedHaloCatalog(simname='bolplanck', version_name='halotools_alpha_version2')
    substr = "See https://github.com/astropy/halotools/issues/598"
    assert substr in err.value.args[0]


def test_lbox_vector():
    """ Ensure that the Lbox attribute of CachedHaloCatalog instances
    is always a 3-element vector.
    """
    for simname in list(adict.keys()):
        alist = adict[simname]
        a = alist[0]
        z = 1/a - 1
        try:
            halocat = CachedHaloCatalog(simname=simname, redshift=z)
            assert len(halocat.Lbox) == 3
        except (InvalidCacheLogEntry, HalotoolsError, AssertionError):
            if APH_MACHINE:
                raise HalotoolsError("APH_MACHINE should never fail Lbox_vector test\n"
                    "simname = {0}\nscale factor = {1}".format(simname, a))
            else:
                pass
