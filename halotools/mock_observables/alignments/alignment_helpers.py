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


def process_projected_alignment_args(sample1, alignments1, ellipticities1, sample2, alignments2, ellipticities2,
                                     rp_bins, pi_max,period, num_threads, approx_cell1_size, approx_cell2_size):
    r"""
    process arguments for  projected alignment correlation functions
    """


