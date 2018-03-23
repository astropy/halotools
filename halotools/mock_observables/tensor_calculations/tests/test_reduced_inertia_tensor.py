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
    concoct a situation where the reduced inertia tensor gives the same result as the stadard.
    """

    Lbox, rsmooth = 1, 0.1
    N1, N2 = 1, 10000

    sample1 = np.random.random((N1, 3))
    sample2 = np.random.random((N2, 3))

    r_12 = np.sqrt((sample1[0,0] - sample2[:,0])**2 + (sample1[0,1] - sample2[:,1])**2 +(sample1[0,2] - sample2[:,2])**2)

    masses2a = np.ones(N2) # equal weight for unreduced tensor
    masses2b = r_12**2 # weights proportional to r^2 in order to cancel 1/r^2 term in the reduced tensor

    #make all particles have same ID
    id1 = np.ones(N1).astype('int')
    id2 = np.ones(N2).astype('int')
    q1 = np.ones(N1)
    s1 = np.ones(N1)

    tensors1, sum_of_masses1 = inertia_tensor_per_object(sample1, sample2, masses2a, rsmooth, period=None)
    tensors2, sum_of_masses2 = reduced_inertia_tensor_per_object(sample1, sample2, rsmooth, masses2b, period=None)
    print(tensors2)
    assert np.allclose(tensors2,tensors1)
