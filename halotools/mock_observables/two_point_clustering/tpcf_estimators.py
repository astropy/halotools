"""
private functions related to estimating the two point correltion function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ...custom_exceptions import HalotoolsError
from ...utils.array_utils import convert_to_ndarray

__all__ = ['_TP_estimator', '_list_estimators', '_TP_estimator_requirements']
__author__ = ['Duncan Campbell']


def _TP_estimator(DD, DR, RR, ND1, ND2, NR1, NR2, estimator):
    """
    two point correlation function estimator
    """

    ND1 = convert_to_ndarray(ND1)
    ND2 = convert_to_ndarray(ND2)
    NR1 = convert_to_ndarray(NR1)
    NR2 = convert_to_ndarray(NR2)
    Ns = np.array([len(ND1), len(ND2), len(NR1), len(NR2)])

    if np.any(Ns>1):
        #used for the jackknife calculations
        #the outer dimension is the number of samples.
        #the N arrays are the number of points in each dimension.
        #so, what we want to do is multiple each row of e.g. DD by the number of 1/N
        mult = lambda x, y: (x*y.T).T  # annoying and ugly, but works.
    else:
        mult = lambda x, y: x*y  # used for all else

    if estimator == 'Natural':
        factor = ND1*ND2/(NR1*NR2)
        #DD/RR-1
        xi = mult(1.0/factor, DD/RR) - 1.0
    elif estimator == 'Davis-Peebles':
        factor = ND1*ND2/(ND1*NR2)
        #DD/DR-1
        xi = mult(1.0/factor, DD/DR) - 1.0
    elif estimator == 'Hewett':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD-DR)/RR
        xi = mult(1.0/factor1, DD/RR) - mult(1.0/factor2, DR/RR)
    elif estimator == 'Hamilton':
        #DDRR/DRDR-1
        xi = (DD*RR)/(DR*DR) - 1.0
    elif estimator == 'Landy-Szalay':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD - 2.0*DR + RR)/RR
        xi = mult(1.0/factor1, DD/RR) - mult(1.0/factor2, 2.0*DR/RR) + 1.0
    else:
        raise ValueError("unsupported estimator!")

    if np.shape(xi)[0]==1: return xi[0]
    else: return xi  # for jackknife


def _list_estimators():
    """
    list available tpcf estimators.
    """
    estimators = ['Natural', 'Davis-Peebles', 'Hewett', 'Hamilton', 'Landy-Szalay']
    return estimators


def _TP_estimator_requirements(estimator):
    """
    return booleans indicating which pairs need to be counted for the chosen estimator
    """
    if estimator == 'Natural':
        do_DD = True
        do_DR = False
        do_RR = True
    elif estimator == 'Davis-Peebles':
        do_DD = True
        do_DR = True
        do_RR = False
    elif estimator == 'Hewett':
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == 'Hamilton':
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == 'Landy-Szalay':
        do_DD = True
        do_DR = True
        do_RR = True
    else:
        available_estimators = _list_estimators()
        if estimator not in available_estimators:
            msg = ("Input `estimator` must be one of the following:{0}".format(available_estimators))
            raise HalotoolsError(msg)

    return do_DD, do_DR, do_RR
