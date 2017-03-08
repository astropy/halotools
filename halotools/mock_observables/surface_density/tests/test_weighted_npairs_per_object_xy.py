"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from .pure_python_weighted_npairs_xy import pure_python_weighted_npairs_xy
from ..weighted_npairs_xy import weighted_npairs_xy
from ..weighted_npairs_per_object_xy import weighted_npairs_per_object_xy

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ('test_weighted_npairs_per_object_xy_brute_force_pbc', )

fixed_seed = 43


def test_weighted_npairs_per_object_xy_brute_force_pbc():
    """
    """
    pass


