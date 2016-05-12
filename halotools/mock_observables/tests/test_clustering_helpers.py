""" Module provides unit-testing for `~halotools.mock_observables.clustering_helpers`. 
"""
from __future__ import absolute_import, division, print_function
from astropy.tests.helper import pytest

from ..clustering_helpers import verify_tpcf_estimator

__all__ = ('test_verify_tpcf_estimator', )

def test_verify_tpcf_estimator():
    """
    """
    _ = verify_tpcf_estimator('Natural')

    with pytest.raises(ValueError) as err:
        _ = verify_tpcf_estimator('Cuba Gooding, Jr.')
    substr = "is not in the list of available estimators:"
    assert substr in err.value.args[0]



