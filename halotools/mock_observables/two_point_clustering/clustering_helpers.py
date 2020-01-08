"""
helper functions used to process arguments passed to the functions in the
`~halotools.mock_observables.two_point_clustering` sub-package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from warnings import warn

from ..mock_observables_helpers import enforce_sample_has_correct_shape

__all__ = ('verify_tpcf_estimator', 'process_optional_input_sample2')

__author__ = ['Duncan Campbell', 'Andrew Hearin']

# Define a dictionary containing the available tpcf estimator names
# and their corresponding values of (do_DD, do_DR, do_RR)
tpcf_estimator_dd_dr_rr_requirements = ({
    'Natural': (True, False, True),
    'Davis-Peebles': (True, True, False),
    'Hewett': (True, True, True),
    'Hamilton': (True, True, True),
    'Landy-Szalay': (True, True, True)
    })


def verify_tpcf_estimator(estimator):
    """ Require that the input ``estimator`` string is one of the available options.

    Parameters
    -----------
    estimator : string

    """
    available_estimators = list(tpcf_estimator_dd_dr_rr_requirements.keys())
    if estimator in available_estimators:
        return estimator
    else:
        msg = (u"Your estimator ``{0}`` \n"
            "is not in the list of available estimators:\n {1}".format(estimator, available_estimators))
        raise ValueError(msg)


def process_optional_input_sample2(sample1, sample2, do_cross, ndim=3):
    """ Function used to process the input ``sample2`` passed to all two-point clustering
    functions in `~halotools.mock_observables`.
    The input ``sample1`` should have already been run through the
    `~halotools.mock_observables.mock_observables_helpers.enforce_sample_has_correct_shape`
    function.

    If the input ``sample2`` is  None, then  `process_optional_input_sample2`
    will set ``sample2`` equal to ``sample1`` and additionally
    return True for ``_sample1_is_sample2``.
    Otherwise, the `process_optional_input_sample2` function
    will verify that the input ``sample2`` has the correct shape.

    The input ``sample2`` will also be tested for equality with ``sample1``.
    If the two samples are equal, the ``_sample1_is_sample2`` will be set to True,
    and ``do_cross`` will be over-written to False.

    Parameters
    ----------
    sample1 : array_like

    sample2 : array_like

    do_cross : boolean

    Returns
    ---------
    sample2 : array_like

    _sample1_is_sample2 : boolean

    do_cross : boolean
    """
    if sample2 is None:
        sample2 = sample1
        _sample1_is_sample2 = True
    else:
        sample2 = enforce_sample_has_correct_shape(sample2, ndim=ndim)
        if sample1.shape != sample2.shape:
            _sample1_is_sample2 = False
        else:
            if np.all(sample1 == sample2):
                _sample1_is_sample2 = True
                if do_cross:
                    msg = (u"\n `sample1` and `sample2` are exactly the same, \n"
                           "only the auto-correlation will be returned.\n")
                    warn(msg)
                    do_cross = False
            else:
                _sample1_is_sample2 = False

    return sample2, _sample1_is_sample2, do_cross
