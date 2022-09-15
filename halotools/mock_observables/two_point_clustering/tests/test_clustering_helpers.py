""" Module provides unit-testing for `~halotools.mock_observables.clustering_helpers`.
"""
from __future__ import absolute_import, division, print_function
import warnings

import pytest
import numpy as np

from ..clustering_helpers import verify_tpcf_estimator
from ..clustering_helpers import process_optional_input_sample2

__all__ = ("test_verify_tpcf_estimator",)


def test_verify_tpcf_estimator():
    """ """
    _ = verify_tpcf_estimator("Natural")

    with pytest.raises(ValueError) as err:
        _ = verify_tpcf_estimator("Cuba Gooding, Jr.")
    substr = "is not in the list of available estimators:"
    assert substr in err.value.args[0]


def test_process_optional_input_sample2_case1():

    npts1, npts2 = 1000, 1000
    sample1_in = np.zeros((npts1, 3))
    sample2_in = None
    do_cross_in = True

    sample2_out, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1_in, sample2_in, do_cross_in
    )
    assert np.all(sample1_in == sample2_out)
    assert _sample1_is_sample2 is True
    assert do_cross == do_cross_in


def test_process_optional_input_sample2_case2():

    npts1, npts2 = 1000, 1000
    sample1_in = np.zeros((npts1, 3))
    sample2_in = np.ones((npts2, 3))

    do_cross_in = False
    sample2_out, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1_in, sample2_in, do_cross_in
    )
    assert np.all(sample2_in == sample2_out)
    assert _sample1_is_sample2 is False
    assert do_cross == do_cross_in

    do_cross_in = True
    sample2_out, _sample1_is_sample2, do_cross = process_optional_input_sample2(
        sample1_in, sample2_in, do_cross_in
    )
    assert do_cross == do_cross_in


def test_process_optional_input_sample2_case3():

    npts1, npts2 = 1000, 1000
    sample1_in = np.zeros((npts1, 3))
    sample2_in = sample1_in

    do_cross_in = True
    with warnings.catch_warnings(record=True) as w:
        sample2_out, _sample1_is_sample2, do_cross = process_optional_input_sample2(
            sample1_in, sample2_in, do_cross_in
        )
        assert "exactly the same" in str(w[-1].message)

    assert np.all(sample2_in == sample2_out)
    assert _sample1_is_sample2 is True
    assert do_cross is False
