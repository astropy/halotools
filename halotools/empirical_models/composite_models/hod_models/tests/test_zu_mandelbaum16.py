"""
"""
import numpy as np
import pytest

from ....factories import PrebuiltHodModelFactory

from .....sim_manager import FakeSim

# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
from astropy.config.paths import _find_home
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ('test_zu_mandelbaum16_correct_fred', )


def test_zu_mandelbaum16_correct_fred():
    halocat = FakeSim()
    model = PrebuiltHodModelFactory('zu_mandelbaum16', threshold=10)
    model.populate_mock(halocat, seed=43)

    # Find some intermediate mass
    mask = halocat.halo_table['halo_mvir'] > 1e13
    mask *= halocat.halo_table['halo_mvir'] <= 1e14
    example_mass = max(set(halocat.halo_table['halo_mvir'][mask]))
    mask = model.mock.galaxy_table['halo_mvir'] == example_mass

    gals = model.mock.galaxy_table[mask]
    cens = gals[gals['gal_type'] == 'centrals']
    sats = gals[gals['gal_type'] == 'satellites']

    expected_red_frac_cens = model.mean_quiescent_fraction_centrals(prim_haloprop=example_mass)
    expected_red_frac_sats = model.mean_quiescent_fraction_satellites(prim_haloprop=example_mass)

    # assert 4 == 5, "len(sats) = {0}, example_mass = {1}".format(len(sats), example_mass)
    assert np.allclose(cens['quiescent'].mean(), expected_red_frac_cens, rtol=0.03)
    assert np.allclose(sats['quiescent'].mean(), expected_red_frac_sats, rtol=0.03)


def test_zu_mandelbaum16_seed():
    halocat = FakeSim()
    model = PrebuiltHodModelFactory('zu_mandelbaum16', threshold=10)

    model.populate_mock(halocat, seed=43)
    ngals1 = len(model.mock.galaxy_table)
    red_cen_mask = model.mock.galaxy_table['gal_type'] == 'centrals'
    red_cen_mask *= model.mock.galaxy_table['quiescent'] == True
    nredcens1 = np.count_nonzero(red_cen_mask)

    model.mock.populate(seed=43)
    ngals2 = len(model.mock.galaxy_table)
    red_cen_mask = model.mock.galaxy_table['gal_type'] == 'centrals'
    red_cen_mask *= model.mock.galaxy_table['quiescent'] == True
    nredcens2 = np.count_nonzero(red_cen_mask)

    model.mock.populate(seed=44)
    ngals3 = len(model.mock.galaxy_table)
    red_cen_mask = model.mock.galaxy_table['gal_type'] == 'centrals'
    red_cen_mask *= model.mock.galaxy_table['quiescent'] == True
    nredcens3 = np.count_nonzero(red_cen_mask)

    assert ngals1 == ngals2
    assert nredcens1 == nredcens2
    assert ngals3 != ngals2
    assert nredcens2 != nredcens3
