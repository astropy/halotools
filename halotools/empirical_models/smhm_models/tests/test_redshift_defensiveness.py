#!/usr/bin/env python

from astropy.tests.helper import pytest

from ...smhm_models import Behroozi10SmHm

from ....sim_manager import sim_defaults
from ....custom_exceptions import HalotoolsError


def test_behroozi10_redshift_safety():
    """
    """
    model = Behroozi10SmHm()

    result0 = model.mean_log_halo_mass(11)
    result1 = model.mean_log_halo_mass(11, redshift=4)
    result2 = model.mean_log_halo_mass(11, redshift=sim_defaults.default_redshift)
    assert result0 == result2
    assert result0 != result1

    result0 = model.mean_stellar_mass(prim_haloprop=1e12)
    result1 = model.mean_stellar_mass(prim_haloprop=1e12, redshift=4)
    result2 = model.mean_stellar_mass(prim_haloprop=1e12, redshift=sim_defaults.default_redshift)
    assert result0 == result2
    assert result0 != result1

    model = Behroozi10SmHm(redshift=sim_defaults.default_redshift)
    result0 = model.mean_log_halo_mass(11)
    with pytest.raises(HalotoolsError) as exc:
        result1 = model.mean_log_halo_mass(11, redshift=4)
    result2 = model.mean_log_halo_mass(11, redshift=model.redshift)
    assert result0 == result2

    result0 = model.mean_stellar_mass(prim_haloprop=1e12)
    with pytest.raises(HalotoolsError) as exc:
        result1 = model.mean_stellar_mass(prim_haloprop=1e12, redshift=4)
    result2 = model.mean_stellar_mass(prim_haloprop=1e12, redshift=model.redshift)
    assert result0 == result2
