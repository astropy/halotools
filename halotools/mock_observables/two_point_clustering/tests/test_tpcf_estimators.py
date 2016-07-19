""" Module provides unit-testing for `~halotools.mock_observables.tpcf_estimators`.
"""
from __future__ import absolute_import, division, print_function

from astropy.tests.helper import pytest
import numpy as np

from ..tpcf_estimators import _test_for_zero_division, _list_estimators

__all__ = ('test_zero_division1', )


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

    RR[0] = 0.
    DR[0] = 0.
    for estimator in _list_estimators():
        with pytest.raises(ValueError) as err:
            _test_for_zero_division(DD, DR, RR, ND1, ND2, NR1, NR2, estimator)
        substr = "you will have at least one NaN returned value"
        assert substr in err.value.args[0]
