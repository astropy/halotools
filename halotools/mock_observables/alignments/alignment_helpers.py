"""
helper functions used to process arguments passed to the functions in the
`~halotools.mock_observables.alignments` sub-package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from warnings import warn

from ..mock_observables_helpers import enforce_sample_has_correct_shape

__all__ = ('process_projected_alignment_args')

__author__ = ['Duncan Campbell']


def process_projected_alignment_args(sample1, orientations1, ellipticities1, weights1,
                                     sample2, orientations2, ellipticities2, weights2):
    r"""
    process arguments for  projected alignment correlation functions
    """

    sample1 = enforce_sample_has_correct_shape(sample1)
    sample2 = enforce_sample_has_correct_shape(sample2)
    N1 = len(sample1)
    N2 = len(sample2)

    orientations1 = np.atleast_1d(orientations1).astype(float)
    ellipticities1 = np.atleast_1d(ellipticities1).astype(float)

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

    if np.shape(orientations1) != (N1, 2):
        msg = ("`orientations1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(orientations2) != (N2, 2):
        msg = ("`orientations2` is not the correct shape.")
        raise ValueError(msg)

    if np.shape(ellipticities1) != (N1,):
        msg = ("`ellipticities1` is not the correct shape.")
        raise ValueError(msg)
    if np.shape(ellipticities2) != (N2,):
        msg = ("`ellipticities2` is not the correct shape.")
        raise ValueError(msg)

    #normalize the orientation vectors
    orientations1 = orientations1/np.sqrt(np.sum(orientations1**2, axis=1)).reshape((len(orientations1), -1))
    orientations2 = orientations2/np.sqrt(np.sum(orientations2**2, axis=1)).reshape((len(orientations2), -1))

    return sample1, orientations1, ellipticities1, weights1, sample2, orientations2, ellipticities2, weights2
