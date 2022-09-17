"""
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from astropy.config.paths import _find_home

from ...factories import PrebuiltHodModelFactory

from ....sim_manager import FakeSim


__all__ = ("test_fake_mock_population", "test_fake_mock_observations1")


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


@pytest.mark.skipif("not APH_MACHINE")
def test_fake_mock_population():
    halocat = FakeSim(num_halos_per_massbin=25)
    for modelname in PrebuiltHodModelFactory.prebuilt_model_nickname_list:
        model = PrebuiltHodModelFactory(modelname)
        model.populate_mock(halocat)
    model.populate_mock(halocat)


@pytest.mark.skipif("not APH_MACHINE")
def test_fake_mock_observations1():
    model = PrebuiltHodModelFactory("zu_mandelbaum16")
    result = model.compute_average_galaxy_clustering(num_iterations=1, simname="fake")

    result = model.compute_average_galaxy_clustering(
        num_iterations=1,
        simname="fake",
        summary_statistic="mean",
        gal_type="centrals",
        include_crosscorr=True,
        rbins=np.array((0.1, 0.2, 0.3)),
        redshift=0,
        halo_finder="rockstar",
    )
