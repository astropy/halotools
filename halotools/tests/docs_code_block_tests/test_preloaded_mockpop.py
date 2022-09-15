""" These tests populate mocks using all the pre-loaded models. These calculations
time-out on the Travis CI environment, and so we run them here on APH_MACHINE only
to ensure the still get executed.
"""
import numpy as np
from astropy.config.paths import _find_home

from ...sim_manager import FakeSim
from ...empirical_models import PrebuiltHodModelFactory, PrebuiltSubhaloModelFactory

# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = "/Users/aphearin"
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


def test_zu_mandelbaum15_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltHodModelFactory("zu_mandelbaum15")

    if APH_MACHINE:
        model.populate_mock(halocat)


def test_zu_mandelbaum16_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltHodModelFactory("zu_mandelbaum16")

    if APH_MACHINE:
        model.populate_mock(halocat)


def test_zheng07_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltHodModelFactory("zheng07")

    if APH_MACHINE:
        model.populate_mock(halocat)


def test_behroozi10_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltSubhaloModelFactory("behroozi10")

    if APH_MACHINE:
        model.populate_mock(halocat)


def test_hearin15_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltHodModelFactory("hearin15")

    if APH_MACHINE:
        model.populate_mock(halocat)


def test_cacciato09_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltHodModelFactory("cacciato09")

    if APH_MACHINE:
        model.populate_mock(halocat)


def test_leauthaud11_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltHodModelFactory("leauthaud11")

    if APH_MACHINE:
        model.populate_mock(halocat)


def test_tinker13_mockpop():

    halocat = FakeSim(seed=43)
    model = PrebuiltHodModelFactory("tinker13")

    if APH_MACHINE:
        model.populate_mock(halocat)
