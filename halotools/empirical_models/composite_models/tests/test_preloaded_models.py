"""
"""
from astropy.tests.helper import pytest

from ...factories import PrebuiltHodModelFactory

from ....sim_manager import CachedHaloCatalog, FakeSim
from ....custom_exceptions import HalotoolsError


__all__ = ('test_hearin15', 'test_Leauthaud11', 'test_Leauthaud11b',
    'test_Leauthaud11c', 'test_zu_mandelbaum15', 'test_zu_mandelbaum15b')


def test_hearin15():
    """
    """
    model = PrebuiltHodModelFactory('hearin15')
    try:
        halocat = CachedHaloCatalog()
    except:
        halocat = FakeSim()
    model.populate_mock(halocat)


def test_Leauthaud11():
    """
    """
    model = PrebuiltHodModelFactory('leauthaud11')
    halocat = FakeSim()
    model.populate_mock(halocat)


def test_Leauthaud11b():
    """
    """
    model = PrebuiltHodModelFactory('leauthaud11')
    halocat = FakeSim(redshift=2.)
    # Test that an attempt to repopulate with a different halocat raises an exception
    with pytest.raises(HalotoolsError) as err:
        model.populate_mock(halocat)  # default redshift != 2
    substr = ""
    assert substr in err.value.args[0]


def test_Leauthaud11c():
    """
    """
    model_highz = PrebuiltHodModelFactory('leauthaud11', redshift=2.)
    halocat = FakeSim(redshift=2.)
    model_highz.populate_mock(halocat)


def test_zu_mandelbaum15b():
    """
    """
    halocat = FakeSim()
    model = PrebuiltHodModelFactory('zu_mandelbaum15', prim_haloprop_key='halo_mvir')
    model.populate_mock(halocat)
