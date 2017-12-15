"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

__all__ = (
        'principal_axes_from_inertia_tensors', 'sphericity_from_inertia_tensors',
        'triaxility_from_inertia_tensors')


def principal_axes_from_inertia_tensors(inertia_tensors):
    evals, evecs = np.linalg.eigh(inertia_tensors)
    return evecs[:, :, 2], evals[:, 2]


def sphericity_from_inertia_tensors(inertia_tensors):
    evals, __ = np.linalg.eigh(inertia_tensors)
    third_evals = evals[:, 0]
    first_evals = evals[:, 2]
    sphericity = third_evals/first_evals
    return sphericity


def triaxility_from_inertia_tensors(inertia_tensors):
    evals, __ = np.linalg.eigh(inertia_tensors)
    third_evals = evals[:, 0]
    second_evals = evals[:, 1]
    first_evals = evals[:, 2]
    triaxility = (first_evals**2 - second_evals**2)/(first_evals**2 - third_evals**2)
    return triaxility
