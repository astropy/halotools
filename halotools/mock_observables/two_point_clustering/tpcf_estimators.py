"""
private functions related to estimating the two point correltion function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ...custom_exceptions import HalotoolsError

__all__ = ["_TP_estimator", "_list_estimators", "_TP_estimator_requirements"]
__author__ = ["Duncan Campbell"]


def _TP_estimator(DD, DR, RR, ND1, ND2, NR1, NR2, estimator):
    """
    two point correlation function estimator
    """

    ND1 = np.atleast_1d(ND1)
    ND2 = np.atleast_1d(ND2)
    NR1 = np.atleast_1d(NR1)
    NR2 = np.atleast_1d(NR2)
    Ns = np.array([len(ND1), len(ND2), len(NR1), len(NR2)])

    if np.any(Ns > 1):
        # used for the jackknife calculations
        # the outer dimension is the number of samples.
        # the N arrays are the number of points in each dimension.
        # so, what we want to do is multiple each row of e.g. DD by the number of 1/N
        def mult(x, y):
            return (x * y.T).T

    else:

        def mult(x, y):
            return x * y

    _test_for_zero_division(DD, DR, RR, ND1, ND2, NR1, NR2, estimator)

    if estimator == "Natural":
        factor = ND1 * ND2 / (NR1 * NR2)
        # DD/RR-1
        xi = mult(1.0 / factor, DD / RR) - 1.0
    elif estimator == "Davis-Peebles":
        factor = ND1 * ND2 / (ND1 * NR2)
        # DD/DR-1
        xi = mult(1.0 / factor, DD / DR) - 1.0
    elif estimator == "Hewett":
        factor1 = ND1 * ND2 / (NR1 * NR2)
        factor2 = ND1 * NR2 / (NR1 * NR2)
        # (DD-DR)/RR
        xi = mult(1.0 / factor1, DD / RR) - mult(1.0 / factor2, DR / RR)
    elif estimator == "Hamilton":
        # DDRR/DRDR-1
        xi = (DD * RR) / (DR * DR) - 1.0
    elif estimator == "Landy-Szalay":
        factor1 = ND1 * ND2 / (NR1 * NR2)
        factor2 = ND1 * NR2 / (NR1 * NR2)
        # (DD - 2.0*DR + RR)/RR
        xi = mult(1.0 / factor1, DD / RR) - mult(1.0 / factor2, 2.0 * DR / RR) + 1.0
    else:
        raise ValueError("unsupported estimator!")

    if np.shape(xi)[0] == 1:
        return xi[0]
    else:
        return xi  # for jackknife


def _TP_estimator_crossx(DD, D1R, D2R, RR, ND1, ND2, NR1, NR2, estimator):
    """
    two point correlation function estimator
    """

    ND1 = np.atleast_1d(ND1)
    ND2 = np.atleast_1d(ND2)
    NR1 = np.atleast_1d(NR1)
    NR2 = np.atleast_1d(NR2)
    Ns = np.array([len(ND1), len(ND2), len(NR1), len(NR2)])

    if np.any(Ns > 1):
        # used for the jackknife calculations
        # the outer dimension is the number of samples.
        # the N arrays are the number of points in each dimension.
        # so, what we want to do is multiple each row of e.g. DD by the number of 1/N
        def mult(x, y):
            return (x * y.T).T

    else:

        def mult(x, y):
            return x * y

    _test_for_zero_division(DD, D1R, RR, ND1, ND2, NR1, NR2, estimator)
    _test_for_zero_division(DD, D2R, RR, ND1, ND2, NR1, NR2, estimator)

    if estimator == "Natural":
        factor = ND1 * ND2 / (NR1 * NR2)
        # DD/RR-1
        xi = mult(1.0 / factor, DD / RR) - 1.0
    elif estimator == "Hamilton":
        # DDRR/DRDR-1
        xi = (DD * RR) / (D1R * D2R) - 1.0
    elif estimator == "Landy-Szalay":
        factor1 = ND1 * ND2 / (NR1 * NR2)
        factor2 = ND1 * NR2 / (NR1 * NR2)
        # (DD - 2.0*DR + RR)/RR
        term1 = mult(1.0 / factor1, DD / RR)
        term2 = mult(1.0 / factor2, D1R / RR)
        term3 = mult(1.0 / factor2, D2R / RR)
        xi = term1 - term2 - term3 + 1.0
    else:
        msg = "{0} estimator is not supported for cross-correlations"
        raise ValueError(msg.format(estimator))

    if np.shape(xi)[0] == 1:
        return xi[0]
    else:
        return xi  # for jackknife


def _list_estimators():
    """
    list available tpcf estimators.
    """
    estimators = ["Natural", "Davis-Peebles", "Hewett", "Hamilton", "Landy-Szalay"]
    return estimators


def _TP_estimator_requirements(estimator):
    """
    return booleans indicating which pairs need to be counted for the chosen estimator
    """
    if estimator == "Natural":
        do_DD = True
        do_DR = False
        do_RR = True
    elif estimator == "Davis-Peebles":
        do_DD = True
        do_DR = True
        do_RR = False
    elif estimator == "Hewett":
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == "Hamilton":
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == "Landy-Szalay":
        do_DD = True
        do_DR = True
        do_RR = True
    else:
        available_estimators = _list_estimators()
        if estimator not in available_estimators:
            msg = "Input `estimator` must be one of the following:{0}".format(
                available_estimators
            )
            raise HalotoolsError(msg)

    return do_DD, do_DR, do_RR


def _test_for_zero_division(DD, DR, RR, ND1, ND2, NR1, NR2, estimator):
    zero_msg = (
        "When calculating the two-point function, there was at least one \n"
        "separation bin with zero {0} pairs. Since the ``{1}`` estimator you chose \n"
        "divides by {0}, you will have at least one NaN returned value.\n"
        "Most likely, the innermost separation bin is the problem.\n"
        "Try increasing the number of randoms and/or using broader bins.\n"
        "To estimate the number of required randoms, the following expression \n"
        "for the expected number of pairs inside a sphere of radius ``r`` may be useful:\n\n"
        "<Npairs> = (Nran_tot)*(4pi/3)*(r/Lbox)^3 \n\n"
    )

    estimators_dividing_by_rr = ("Natural", "Davis-Peebles", "Hewett", "Landy-Szalay")
    if (estimator in estimators_dividing_by_rr) & (np.any(RR == 0)):
        raise ValueError(zero_msg.format("RR", estimator))

    estimators_dividing_by_dr = ("Hamilton",)
    if (estimator in estimators_dividing_by_dr) & (np.any(DR == 0)):
        raise ValueError(zero_msg.format("DR", estimator))
