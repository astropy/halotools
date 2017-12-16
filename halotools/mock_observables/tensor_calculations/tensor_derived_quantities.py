"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

__all__ = (
        'principal_axes_from_inertia_tensors', 'sphericity_from_inertia_tensors',
        'triaxility_from_inertia_tensors')


def principal_axes_from_inertia_tensors(inertia_tensors):
    r""" Calculate the principal eigenvector of each of the input inertia tensors.

    Parameters
    ----------
    inertia_tensors : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of 3x3 symmetric
        positive-definite matrices

    Returns
    -------
    principal_axes : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d principal eigenvectors

    eigenvalues : ndarray
        Numpy array of shape (npts, ) storing the eigenvalue of each principal eigenvector

    Notes
    -----
    The function `~halotools.mock_observables.inertia_tensor_per_object`
    calculates the inertia tensors :math:`\mathcal{I}_{\rm i}` for a collection of
    points inside a 3d mass distribution.

    """
    evals, evecs = np.linalg.eigh(inertia_tensors)
    return evecs[:, :, 2], evals[:, 2]


def sphericity_from_inertia_tensors(inertia_tensors):
    r""" Calculate the sphericity :math:`\mathcal{S}_{\rm i}`
    of each of the :math:`i=1,\dots,N_{\rm points}`
    mass distributions defined by the input inertia tensors :math:`\mathcal{I}_{\rm i}`.

    Sphericity :math:`0 < \mathcal{S} <= 1` is defined in terms of
    the eigenvalues of the inertia tensor :math:`\mathcal{I}`,
    denoted by :math:`\lambda_{a}, \lambda_{b}, \lambda_{c}`,
    from largest to smallest:

    .. math::
        \mathcal{S}\equiv\lambda_{c}/\lambda_{a}

    Parameters
    ----------
    inertia_tensors : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of 3x3 symmetric
        positive-definite matrices

    Returns
    -------
    sphericity : ndarray
        Numpy array of shape (npts, ) storing the sphericity of each inertia tensor

    Notes
    -----
    The function `~halotools.mock_observables.inertia_tensor_per_object`
    calculates the inertia tensors :math:`\mathcal{I}_{\rm i}` for a collection of
    points inside a 3d mass distribution.
    """
    evals, evecs = np.linalg.eigh(inertia_tensors)
    third_evals = evals[:, 0]
    first_evals = evals[:, 2]
    sphericity = third_evals/first_evals
    return sphericity


def triaxility_from_inertia_tensors(inertia_tensors):
    r""" Calculate the triaxility :math:`\mathcal{T}_{\rm i}`
    of each of the :math:`i=1,\dots,N_{\rm points}`
    mass distributions defined by the input inertia tensors :math:`\mathcal{I}_{\rm i}`.

    Triaxility :math:`\mathcal{T}` is defined in terms of
    the eigenvalues of the inertia tensor :math:`\mathcal{I}`,
    denoted by :math:`\lambda_{a}, \lambda_{b}, \lambda_{c}`,
    from largest to smallest:

    .. math::
        \mathcal{T}\equiv\frac{\lambda_{a}^{2}-\lambda_{b}^{2}}{\lambda_{a}^{2}-\lambda_{c}^{2}}


    Parameters
    ----------
    inertia_tensors : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of 3x3 symmetric
        positive-definite matrices

    Returns
    -------
    triaxility : ndarray
        Numpy array of shape (npts, ) storing the triaxility of each inertia tensor

    Notes
    -----
    The function `~halotools.mock_observables.inertia_tensor_per_object`
    calculates the inertia tensors :math:`\mathcal{I}_{\rm i}` for a collection of
    points inside a 3d mass distribution.
    """
    evals, evecs = np.linalg.eigh(inertia_tensors)
    third_evals = evals[:, 0]
    second_evals = evals[:, 1]
    first_evals = evals[:, 2]
    triaxility = (first_evals**2 - second_evals**2)/(first_evals**2 - third_evals**2)
    return triaxility
