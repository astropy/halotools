"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..inertia_tensor import inertia_tensor_per_object
from ..reduced_inertia_tensor import reduced_inertia_tensor_per_object

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh


__all__ = ('test_1', )
fixed_seed = 43

def test_1():
    """
    """

    Lbox, rsmooth = 250., 5.
    N1, N2 = 100, 1000

    sample1 = np.random.random((N1, 3))
    sample2 = np.random.random((N2, 3))

    masses = np.random.random(N2)

    #make all particles have same ID
    id1 = np.ones(N1).astype('int')
    id2 = np.ones(N2).astype('int')
    q1 = np.ones(N1)
    s1 = np.ones(N1)

    tensors1, sum_of_masses1 = inertia_tensor_per_object(sample1, sample2, masses, rsmooth, period=Lbox)
    tensors2, sum_of_masses2 = reduced_inertia_tensor_per_object(sample1, sample2, masses, rsmooth, id1, id2, q1, s1, period=Lbox)

    print(tensors1[0])
    print(tensors2[0])
    assert True==False
    assert np.all(tensors2==tensors1)
