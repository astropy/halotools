"""
"""
import pytest
import warnings
from ...smhm_models import Behroozi10SmHm

from ....sim_manager import sim_defaults
from ....custom_exceptions import HalotoolsError


def test_behroozi10_redshift_safety():
    """ """
    model = Behroozi10SmHm()

    with warnings.catch_warnings(record=True) as w:
        result0 = model.mean_log_halo_mass(11)
        assert "default_redshift" in str(w[-1].message)
    result1 = model.mean_log_halo_mass(11, redshift=4)
    result2 = model.mean_log_halo_mass(11, redshift=sim_defaults.default_redshift)

    assert result0 == result2
    assert result0 != result1

    with warnings.catch_warnings(record=True) as w:
        result0 = model.mean_stellar_mass(prim_haloprop=1e12)
        assert "default_redshift" in str(w[-1].message)
    result1 = model.mean_stellar_mass(prim_haloprop=1e12, redshift=4)
    result2 = model.mean_stellar_mass(
        prim_haloprop=1e12, redshift=sim_defaults.default_redshift
    )
    assert result0 == result2
    assert result0 != result1

    model = Behroozi10SmHm(redshift=sim_defaults.default_redshift)
    result0 = model.mean_log_halo_mass(11)
    with pytest.raises(HalotoolsError):
        result1 = model.mean_log_halo_mass(11, redshift=4)
    result2 = model.mean_log_halo_mass(11, redshift=model.redshift)
    assert result0 == result2

    result0 = model.mean_stellar_mass(prim_haloprop=1e12)
    with pytest.raises(HalotoolsError):
        result1 = model.mean_stellar_mass(prim_haloprop=1e12, redshift=4)
    result2 = model.mean_stellar_mass(prim_haloprop=1e12, redshift=model.redshift)
    assert result0 == result2
