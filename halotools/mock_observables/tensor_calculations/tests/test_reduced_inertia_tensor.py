"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..inertia_tensor import inertia_tensor_per_object
from ..reduced_inertia_tensor import reduced_inertia_tensor_per_object

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh
from ..tensor_derived_quantities import eigenvectors, axis_ratios_from_inertia_tensors
from ....utils import rotation_matrices_from_angles, rotate_vector_collection


__all__ = ('test_reduced_inertia_tensor1', 'test_reduced_inertia_tensor2', 'test_reduced_inertia_tensor3', 'test_reduced_inertia_tensor4')
fixed_seed = 43


def test_reduced_inertia_tensor1():
    """
    calculate the inertia tensor and the reduced inertia tensor.  
    For the reduced inertia tensor, weight each point by r^2
    so that the result should be exactly the same as the inertia tensor result.
    """

    Lbox, rsmooth = 1, 0.1
    N1, N2 = 1, 10000

    sample1 = np.random.random((N1, 3))
    sample2 = np.random.random((N2, 3))

    r_12 = np.sqrt((sample1[0,0] - sample2[:,0])**2 + (sample1[0,1] - sample2[:,1])**2 +(sample1[0,2] - sample2[:,2])**2)

    # equal weight for unreduced tensor
    masses2a = np.ones(N2)
    # weights proportional to r^2 in order to cancel 1/r^2 term in the reduced tensor
    masses2b = r_12**2

    tensors1, sum_of_masses1 = inertia_tensor_per_object(sample1, sample2, rsmooth, weights2=masses2a, period=None)
    tensors2, sum_of_masses2 = reduced_inertia_tensor_per_object(sample1, sample2, rsmooth, masses2b, period=None)
    
    assert np.allclose(tensors2, tensors1)


def test_reduced_inertia_tensor2():
    """
    recover a known inertia tensor for an ellipsoidal distribution
    with eigenvectors aligned the the x,y,z-axes
    """

    Lbox, rsmooth = 1, 0.3
    N1, N2 = 1, 50000

    # define axis ratios of ellipsoidal distribution
    q0 = 0.8
    s0 = 0.4

    # define center of desitribution
    sample1 = np.array([[0.5,0.5,0.5]])
    # sample an ellipsoidal surface
    sample2 = sample_ellipsoidal_surface(N2, q0, s0)*0.05 + 0.5
    
    # give equal weight to each point
    masses2 = np.ones(N2)

    # give the known inertia tensor as the intial guess
    tensor_0 = np.array([[[1,0,0],[0,q0**2,0],[0,0,s0**2]]])
    tensors_1, sum_of_masses1_1 = reduced_inertia_tensor_per_object(sample1, sample2, rsmooth, period=Lbox, inertia_tensor_0=tensor_0)
    
    # extract the axis ratios
    q1, s1 = axis_ratios_from_inertia_tensors(tensors_1)

    # enforce that the calculated axis ratios are close the the known result
    assert np.fabs(q0 - q1)/q0 < 0.01, print("q= ", q0, q1[0])
    assert np.fabs(s0 - s1)/s0 < 0.01, print("s= ", s0, s1[0])


def test_reduced_inertia_tensor3():
    """
    recover a known inertia tensor for an ellipsoidal distribution
    with eigenvectors mis-aligned the the x,y,z-axes
    """

    Lbox, rsmooth = 1, 0.3
    N1, N2 = 1, 50000

    q0 = 0.8
    s0 = 0.4

    # define center of desitribution
    sample1 = np.array([[0.5,0.5,0.5]])
    # sample an ellipsoidal surface with the major (minor) axis aligned the the z-axis (x-axis)
    sample2 = sample_ellipsoidal_surface(N2, q0, s0, [0,1,0], np.pi/2.0)*0.05 + 0.5

    # give equal weight to each point
    masses2 = np.ones(N2)

    tensor_0 = np.array([[[s0**2,0,0],[0,q0**2,0],[0,0,1]]])
    tensors_1, sum_of_masses1_1 = reduced_inertia_tensor_per_object(sample1, sample2, rsmooth, period=Lbox, inertia_tensor_0=tensor_0)
    
    q1, s1 = axis_ratios_from_inertia_tensors(tensors_1)

    # enforce that the calculated axis ratios are close the the known result
    assert np.fabs(q0 - q1)/q0 < 0.01, print("q= ", q0, q1[0])
    assert np.fabs(s0 - s1)/s0 < 0.01, print("s= ", s0, s1[0])


def test_reduced_inertia_tensor4():
    """
    recover a known inertia tensor for two ellipsoidal distributions
    with eigenvectors aligned the the x,y,z-axes
    """

    Lbox, rsmooth = 1, 0.3
    N1, N2 = 2, 100000

    q0a = 0.8
    s0a = 0.4

    q0b = 0.9
    s0b = 0.5

    sample1 = np.array([0.5,0.5,0.5]*2).reshape((2,3))
    sample2a = sample_ellipsoidal_surface(int(N2/2), q0a, s0a)*0.05 + 0.5
    sample2b = sample_ellipsoidal_surface(int(N2/2), q0b, s0b)*0.05 + 0.5
    sample2 = np.vstack((sample2a, sample2b))

    masses2 = np.ones(N2)

    # make all particles have same ID
    id1 = np.ones(N1).astype('int')
    id1[1] = 2
    id2 = np.ones(N2).astype('int')
    id2[int(N2/2):] = 2

    tensor_0 = np.array([[[1,0,0],[0,q0a**2,0],[0,0,s0a**2]], [[1,0,0],[0,q0b**2,0],[0,0,s0b**2]]])
    tensors_1, sum_of_masses1_1 = reduced_inertia_tensor_per_object(sample1, sample2, rsmooth, id1=id1, id2=id2, period=Lbox, inertia_tensor_0=tensor_0)
    
    q1, s1 = axis_ratios_from_inertia_tensors(tensors_1)
    q1a = q1[0]
    q1b = q1[1]
    s1a = s1[0]
    s1b = s1[1]

    assert np.fabs(q0a - q1a)/q0a < 0.01
    assert np.fabs(s0a - s1a)/s0a < 0.01
    assert np.fabs(q0b - q1b)/q0b < 0.01
    assert np.fabs(s0b - s1b)/s0b < 0.01


def sample_ellipsoidal_surface(N, q=1, s=1, rotation_axis=None, rotation_angle=None):
    """
    randomly sample an ellipsoidal surface with the major (minor) axis initially 
    aligned the the x-axis (z-axis), thn rotate the distribution around the
    specified axis of rotaiton. 

    Parameters
    ----------
    N : int
        number of points

    q : float, optional
        intermediate axis ratio, b/a

    s : float, optional
        minor axis ratio, c/a

    rotation_axis : array_like, optional
        shape (3,) numpy array indicating rotaiton axis.  Default is
        the x-axis

    rotation_angle : float
        angle of rotaiton in radians. Default is 0 rad.

    Returns 
    -------
    coords :  
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

    coords = np.vstack((x,y,z)).T
    
    if rotation_angle is None:
        rotation_angle = 0.0
    if rotation_axis is None:
        rotation_axis = np.array([1, 0, 0])
    
    rot_m = rotation_matrices_from_angles(rotation_angle, rotation_axis)
    coords = rotate_vector_collection(rot_m, coords)

    return coords


