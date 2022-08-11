""" Module provides unit-testing for `~halotools.mock_observables.tpcf_estimators`.
"""
from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from ..tpcf_estimators import (
    _test_for_zero_division,
    _list_estimators,
    _TP_estimator_requirements,
)
from ....custom_exceptions import HalotoolsError

__all__ = ("test_zero_division1",)


def test_zero_division1():
    nbins = 10
    DD = np.arange(nbins) + 10
    DR = np.arange(nbins) + 10
    RR = np.arange(nbins) + 10
    ND1, ND2, NR1, NR2 = 100, 100, 100, 100

    for estimator in _list_estimators():
        _test_for_zero_division(DD, DR, RR, ND1, ND2, NR1, NR2, estimator)


def test_zero_division2():
    nbins = 10
    DD = np.arange(nbins) + 10
    DR = np.arange(nbins) + 10
    RR = np.arange(nbins) + 10
    ND1, ND2, NR1, NR2 = 100, 100, 100, 100

    RR[0] = 0.0
    DR[0] = 0.0
    for estimator in _list_estimators():
        with pytest.raises(ValueError) as err:
            _test_for_zero_division(DD, DR, RR, ND1, ND2, NR1, NR2, estimator)
        substr = "you will have at least one NaN returned value"
        assert substr in err.value.args[0]


def test_TP_estimator_requirements_davis_peebles():
    do_DD, do_DR, do_RR = _TP_estimator_requirements("Davis-Peebles")
    assert np.all((do_DD, do_DR, do_RR) == (True, True, False))


def test_TP_estimator_requirements_hewett():
    do_DD, do_DR, do_RR = _TP_estimator_requirements("Hewett")
    assert np.all((do_DD, do_DR, do_RR) == (True, True, True))


def test_TP_estimator_requirements_hamilton():
    do_DD, do_DR, do_RR = _TP_estimator_requirements("Hamilton")
    assert np.all((do_DD, do_DR, do_RR) == (True, True, True))


def test_TP_estimator_requirements_bad_estimator():
    with pytest.raises(HalotoolsError) as err:
        __ = _TP_estimator_requirements("Ron Perlman")
    substr = "Input `estimator` must be one of the following"
    assert substr in err.value.args[0]
