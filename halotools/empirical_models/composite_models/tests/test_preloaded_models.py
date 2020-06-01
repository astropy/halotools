"""
"""
import sys
import pytest
import numpy as np

from ...factories import PrebuiltHodModelFactory

from ....sim_manager import CachedHaloCatalog, FakeSim
from ....custom_exceptions import HalotoolsError
from ....utils.python_string_comparisons import compare_strings_py23_safe


__all__ = ('test_hearin15', )



@pytest.mark.skipif(sys.platform == 'win32',
                    reason="does not run on windows")
def test_memory_leak():
    model = PrebuiltHodModelFactory('hearin15')
    halocat = FakeSim()
    import resource

    model.populate_mock(halocat)
    maxrss = []
    for i in range(9):
        model.mock.populate()
        maxrss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    import numpy
    # memory usage shall not increase significantly per run.
    assert (numpy.diff(maxrss) < 1024).all()


def test_hearin15():
    """
    """
    model = PrebuiltHodModelFactory('hearin15')
    try:
        halocat = CachedHaloCatalog()
    except:
        halocat = FakeSim()
    model.populate_mock(halocat)


@pytest.mark.installation_test
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


def test_zu_mandelbaum15():
    """
    """
    halocat = FakeSim()
    model = PrebuiltHodModelFactory('zu_mandelbaum15', prim_haloprop_key='halo_mvir')
    model.populate_mock(halocat)


def test_zheng07_alternate_haloprop1():
    """ Regression test for Issue #827 - https://github.com/astropy/halotools/issues/827
    """
    model = PrebuiltHodModelFactory('zheng07', prim_haloprop_key='halo_custom_mass', mdef='200c')

    centrals_occupation = model.model_dictionary['centrals_occupation']
    assert compare_strings_py23_safe(centrals_occupation.prim_haloprop_key, 'halo_custom_mass')

    satellites_occupation = model.model_dictionary['satellites_occupation']
    assert compare_strings_py23_safe(satellites_occupation.prim_haloprop_key, 'halo_custom_mass')

    centrals_profile = model.model_dictionary['centrals_profile']
    assert compare_strings_py23_safe(centrals_profile.mdef, '200c')

    satellites_profile = model.model_dictionary['satellites_profile']
    assert compare_strings_py23_safe(satellites_profile.mdef, '200c')


def test_zheng07_alternate_haloprop2():
    """ Regression test for Issue #827 - https://github.com/astropy/halotools/issues/827
    """
    model = PrebuiltHodModelFactory('zheng07', prim_haloprop_key='halo_custom_mass', mdef='200c')
    halocat = FakeSim(redshift=0.)
    halocat.halo_table['halo_custom_mass'] = np.copy(halocat.halo_table['halo_mvir'])
    halocat.halo_table['halo_m200c'] = np.copy(halocat.halo_table['halo_mvir'])
    halocat.halo_table['halo_r200c'] = np.copy(halocat.halo_table['halo_rvir'])
    model.populate_mock(halocat)

