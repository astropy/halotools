"""
helper functions used to process arguments passed to the functions in this package

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from warnings import warn

from ..mock_observables_helpers import enforce_sample_has_correct_shape

__all__ = ('process_projected_alignment_args', 'process_3d_alignment_args')

__author__ = ['Duncan Campbell']


def process_projected_alignment_args(sample1, orientations1, ellipticities1, weights1,
                                     sample2, orientations2, ellipticities2, weights2,
                                     randoms1, ran_weights1, randoms2, ran_weights2):
    r"""
    process arguments for projected alignment correlation functions
    """

    sample1 = enforce_sample_has_correct_shape(sample1)
    sample2 = enforce_sample_has_correct_shape(sample2)
    N1 = len(sample1)
    N2 = len(sample2)

    # determine if real randoms are passed
    if (randoms1 is not None) and (randoms2 is not None):
        using_randoms = True
        randoms1 = enforce_sample_has_correct_shape(randoms1)
        randoms2 = enforce_sample_has_correct_shape(randoms2)
        NR1 = len(randoms1)
        NR2 = len(randoms2)
    else:
        using_randoms = False
        ran_weights1 = None
        ran_weights2 = None

    # check to see if orientations and ellipticities were provided for sample1
    if orientations1 is not None:
        orientations1 = np.atleast_1d(orientations1).astype(float)
    else:
        orientations1 = np.ones((N1, 2))
    if ellipticities1 is not None:
        ellipticities1 = np.atleast_1d(ellipticities1).astype(float)
    else:
        ellipticities1 = np.ones(N1)

    # check to see if orientations and ellipticities were provided for sample2
    if orientations2 is not None:
        orientations2 = np.atleast_1d(orientations2).astype(float)
    else:
        orientations2 = np.ones((N2, 2))
    if ellipticities2 is not None:
        ellipticities2 = np.atleast_1d(ellipticities2).astype(float)
    else:
        ellipticities2 = np.ones(N2)

    # process weights argument
    if weights1 is not None:
        weights1 = np.atleast_1d(weights1).astype(float)
    else:
        weights1 = np.ones(len(sample1)).astype(float)

    if weights2 is not None:
        weights2 = np.atleast_1d(weights2).astype(float)
    else:
        weights2 = np.ones(len(sample2)).astype(float)

    if using_randoms:
        if ran_weights1 is not None:
            ran_weights1 = np.atleast_1d(ran_weights1).astype(float)
        else:
            ran_weights1 = np.ones(len(randoms1)).astype(float)
        if ran_weights2 is not None:
            ran_weights2 = np.atleast_1d(ran_weights2).astype(float)
        else:
            ran_weights2 = np.ones(len(randoms2)).astype(float)

    # process orientations
    if np.shape(orientations1) != (N1, 2):
        msg = ("`orientations1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(orientations2) != (N2, 2):
        msg = ("`orientations2` is not the correct shape.")
        raise ValueError(msg)

    # process ellipticities
    if np.shape(ellipticities1) != (N1,):
        msg = ("`ellipticities1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(ellipticities2) != (N2,):
        msg = ("`ellipticities2` is not the correct shape.")
        raise ValueError(msg)
    
    # check to make sure weights are correct shape
    if np.shape(weights1) != (N1,):
        msg = ("`weights1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(weights2) != (N2,):
        msg = ("`weights2` is not the correct shape.")
        raise ValueError(msg)

    if using_randoms:
        if np.shape(ran_weights1) != (NR1,):
            msg = ("`ran_weights1` is not the correct shape.")
            raise ValueError(msg)
        if np.shape(ran_weights2) != (NR2,):
            msg = ("`ran_weights1` is not the correct shape.")
            raise ValueError(msg)

    # make sure neither orientations contain a vector of length 0
    mag1 = np.sqrt(np.sum(orientations1**2, axis=1)).reshape((len(orientations1), -1))
    mag2 = np.sqrt(np.sum(orientations2**2, axis=1)).reshape((len(orientations2), -1))

    if np.any(mag1 == 0.0) | np.any(mag2 == 0.0):
        msg = ("`orientations1` or `orientations2` contains a vector of length 0.")
        raise ValueError(msg)

    # normalize the orientation vectors
    orientations1 = orientations1/mag1
    orientations2 = orientations2/mag2

    return sample1, orientations1, ellipticities1, weights1,\
        sample2, orientations2, ellipticities2, weights2,\
        randoms1, ran_weights1, randoms2, ran_weights2


def process_3d_alignment_args(sample1, orientations1, ellipticities1, weights1,
                              sample2, orientations2, ellipticities2, weights2,
                              randoms1, ran_weights1, randoms2, ran_weights2):
    r"""
    process arguments for 3D alignment correlation functions
    """

    sample1 = enforce_sample_has_correct_shape(sample1)
    sample2 = enforce_sample_has_correct_shape(sample2)
    N1 = len(sample1)
    N2 = len(sample2)

    # determine if real randoms are passed
    if (randoms1 is not None) and (randoms2 is not None):
        using_randoms = True
        randoms1 = enforce_sample_has_correct_shape(randoms1)
        randoms2 = enforce_sample_has_correct_shape(randoms2)
        NR1 = len(randoms1)
        NR2 = len(randoms2)
    else:
        using_randoms = False
        ran_weights1 = None
        ran_weights2 = None

    # check to see if orientations and ellipticities were provided for sample1
    if orientations1 is not None:
        orientations1 = np.atleast_1d(orientations1).astype(float)
    else:
        orientations1 = np.ones((N1, 3))
    if ellipticities1 is not None:
        ellipticities1 = np.atleast_1d(ellipticities1).astype(float)
    else:
        ellipticities1 = np.ones(N1)

    # check to see if orientations and ellipticities were provided for sample2
    if orientations2 is not None:
        orientations2 = np.atleast_1d(orientations2).astype(float)
    else:
        orientations2 = np.ones((N2, 3))
    if ellipticities2 is not None:
        ellipticities2 = np.atleast_1d(ellipticities2).astype(float)
    else:
        ellipticities2 = np.ones(N2)

    # process weights argument
    if weights1 is not None:
        weights1 = np.atleast_1d(weights1).astype(float)
    else:
        weights1 = np.ones(len(sample1)).astype(float)

    if weights2 is not None:
        weights2 = np.atleast_1d(weights2).astype(float)
    else:
        weights2 = np.ones(len(sample2)).astype(float)

    if using_randoms:
        if ran_weights1 is not None:
            ran_weights1 = np.atleast_1d(ran_weights1).astype(float)
        else:
            ran_weights1 = np.ones(len(randoms1)).astype(float)
        if ran_weights2 is not None:
            ran_weights2 = np.atleast_1d(ran_weights2).astype(float)
        else:
            ran_weights2 = np.ones(len(randoms2)).astype(float)

    # process orientations
    if np.shape(orientations1) != (N1, 3):
        msg = ("`orientations1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(orientations2) != (N2, 3):
        msg = ("`orientations2` is not the correct shape.")
        raise ValueError(msg)

    # process ellipticities
    if np.shape(ellipticities1) != (N1,):
        msg = ("`ellipticities1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(ellipticities2) != (N2,):
        msg = ("`ellipticities2` is not the correct shape.")
        raise ValueError(msg)
    
    # check to make sure weights are correct shape
    if np.shape(weights1) != (N1,):
        msg = ("`weights1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(weights2) != (N2,):
        msg = ("`weights1` is not the correct shape.")
        raise ValueError(msg)

    if using_randoms:
        if np.shape(ran_weights1) != (NR1,):
            msg = ("`ran_weights1` is not the correct shape.")
            raise ValueError(msg)
        if np.shape(ran_weights2) != (NR2,):
            msg = ("`ran_weights2` is not the correct shape.")
            raise ValueError(msg)

    # make sure neither orientations contain a vector of length 0
    mag1 = np.sqrt(np.sum(orientations1**2, axis=1)).reshape((len(orientations1), -1))
    mag2 = np.sqrt(np.sum(orientations2**2, axis=1)).reshape((len(orientations2), -1))

    if np.any(mag1 == 0.0) | np.any(mag2 == 0.0):
        msg = ("`orientations1` or `orientations2` contains a vector of length 0.")
        raise ValueError(msg)

    # normalize the orientation vectors
    orientations1 = orientations1/mag1
    orientations2 = orientations2/mag2

    return sample1, orientations1, ellipticities1, weights1,\
        sample2, orientations2, ellipticities2, weights2,\
        randoms1, ran_weights1, randoms2, ran_weights2
