"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..inertia_tensor import inertia_tensor_per_object
from ..reduced_inertia_tensor import reduced_inertia_tensor_per_object

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh
from ..tensor_derived_quantities import eigenvectors, axis_ratios_from_inertia_tensors


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
    
    assert np.allclose(tensors2,tensors1)

def test_2():
    """
    recover known inertia tensor
    """

    Lbox, rsmooth = 100, 30
    N1, N2 = 1, 100000

    q = 0.8
    s = 0.4

    sample1 = np.array([[50,50,50]])
    sample2 = sample_ellipsoid_xyz(N2, q, s) + 50

    masses2 = np.ones(N2)

    #make all particles have same ID
    id1 = np.ones(N1).astype('int')
    id2 = np.ones(N2).astype('int')

    tensor_0 = np.array([[[1,0,0],[0,q**2,0],[0,0,s**2]]])
    tensors_1, sum_of_masses1_1 = reduced_inertia_tensor_per_object(sample1, sample2, rsmooth, period=Lbox, inertia_tensor_0=tensor_0)
    
    q1, s1 = axis_ratios_from_inertia_tensors(tensors_1)

    assert np.fabs(q - q1)/q < 0.01
    assert np.fabs(s - s1)/s < 0.01


def test_3():
    """
    recover know inertia tensor with axis rotation
    """

    Lbox, rsmooth = 100, 30
    N1, N2 = 1, 100000

    q = 0.8
    s = 0.4

    sample1 = np.array([[50,50,50]])
    sample2 = sample_ellipsoid_zyx(N2, q, s) + 50

    masses2 = np.ones(N2)

    #make all particles have same ID
    id1 = np.ones(N1).astype('int')
    id2 = np.ones(N2).astype('int')

    tensor_0 = np.array([[[s**2,0,0],[0,q**2,0],[0,0,1]]])
    tensors_1, sum_of_masses1_1 = reduced_inertia_tensor_per_object(sample1, sample2, rsmooth, period=Lbox, inertia_tensor_0=tensor_0)
    
    q1, s1 = axis_ratios_from_inertia_tensors(tensors_1)

    assert np.fabs(q - q1)/q < 0.01
    assert np.fabs(s - s1)/s < 0.01


def sample_ellipsoid_xyz(N, q=1, s=1):
    """
    sample ellipsodial surface with major axis aligned with the x-axis
    """
    phi = np.random.uniform(0, 2*np.pi, N)
    uran = np.random.rand(N)*2 - 1

    cos_t = uran
    sin_t = np.sqrt((1.-cos_t*cos_t))

    c_to_a = s
    c_to_b = s/q

    # temporarily use x-axis as the major axis
    x = 1.0/c_to_a*sin_t * np.cos(phi)
    y = 1.0/c_to_b*sin_t * np.sin(phi)
    z = cos_t

    return np.vstack((x,y,z)).T


def sample_ellipsoid_zyx(N, q=1, s=1):
    """
    sample ellipsodial surface with major axis aligned with the z-axis
    """
    phi = np.random.uniform(0, 2*np.pi, N)
    uran = np.random.rand(N)*2 - 1

    cos_t = uran
    sin_t = np.sqrt((1.-cos_t*cos_t))

    c_to_a = s
    c_to_b = s/q

    # temporarily use x-axis as the major axis
    x = 1.0/c_to_a*sin_t * np.cos(phi)
    y = 1.0/c_to_b*sin_t * np.sin(phi)
    z = cos_t

    return np.vstack((z,y,x)).T


