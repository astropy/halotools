"""
helper functions used to process arguments passed to the functions in the 
`~halotools.mock_observables.two_point_clustering` sub-package.  
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from warnings import warn
from multiprocessing import cpu_count 

from .tpcf_estimators import _list_estimators

from ..mock_observables_helpers import enforce_sample_has_correct_shape

from ...custom_exceptions import HalotoolsError
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ('_tpcf_one_two_halo_decomp_process_args')

__author__=['Duncan Campbell', 'Andrew Hearin']

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
    available_estimators = list(tpcf_estimator_dd_dr_rr_requirements.keys())
    if estimator in available_estimators:
        return estimator
    else:
        msg = ("Your estimator ``{0}`` \n"
            "is not in the list of available estimators:\n {1}".format(estimator, available_estimators))
        raise ValueError(msg)


def process_optional_input_sample2(sample1, sample2, do_cross):
    """ Function used to process the input ``sample2`` passed to all two-point clustering 
    functions in `~halotools.mock_observables`. The input ``sample1`` should have already 
    been run through the 
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
    """
    if sample2 is None:
        sample2 = sample1
        _sample1_is_sample2 = True
    else:
        sample2 = enforce_sample_has_correct_shape(sample2)
        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("\n `sample1` and `sample2` are exactly the same, \n"
                   "only the auto-correlation will be returned.\n")
            warn(msg)
            do_cross = False
        else: 
            _sample1_is_sample2 = False

    return sample2, _sample1_is_sample2, do_cross


def downsample_inputs_exceeding_max_sample_size(sample1, sample2, _sample1_is_sample2, max_sample_size):
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            msg = ("\n `sample1` exceeds `max_sample_size` \n"
                   "downsampling `sample1`...")
            warn(msg)
        else:
            pass
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            msg = ("\n `sample1` exceeds `max_sample_size` \n"
                   "downsampling `sample1`...")
            warn(msg)
        else:
            pass
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            msg = ("\n `sample2` exceeds `max_sample_size` \n"
                   "downsampling `sample2`...")
            warn(msg)
        else:
            pass

    return sample1, sample2



def _tpcf_one_two_halo_decomp_process_args(sample1, sample1_host_halo_id, rbins,
    sample2, sample2_host_halo_id, randoms,
    period, do_auto, do_cross, estimator,num_threads, max_sample_size, 
    approx_cell1_size, approx_cell2_size,approx_cellran_size):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.tpcf_one_two_halo_decomp`. 
    """
    
    sample1 = convert_to_ndarray(sample1)
    sample1_host_halo_id = convert_to_ndarray(sample1_host_halo_id, dt = np.int64)
    
    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)
        sample2_host_halo_id = convert_to_ndarray(sample2_host_halo_id, dt = np.int64)
        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("\n `sample1` and `sample2` are exactly the same, \n"
                   "only the auto-correlation will be returned.\n")
            warn(msg)
            do_cross=False
        else: 
            _sample1_is_sample2 = False
    else: 
        sample2 = sample1
        sample2_host_halo_id = sample1_host_halo_id
        _sample1_is_sample2 = True
    
    if randoms is not None: 
        randoms = convert_to_ndarray(randoms)
        
    #test to see if halo ids are the same length as samples
    if np.shape(sample1_host_halo_id) != (len(sample1),):
        msg = ("\n `sample1_host_halo_id` must be a 1-D \n"
               "array the same length as `sample1`.")
        raise HalotoolsError(msg)
    if np.shape(sample2_host_halo_id) != (len(sample2),):
        msg = ("\n `sample2_host_halo_id` must be a 1-D \n"
               "array the same length as `sample2`.")
        raise HalotoolsError(msg)
    
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            msg = ("\n `sample1` exceeds `max_sample_size` \n"
                   "downsampling `sample1`...")
            warn(msg)
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            msg = ("\n `sample1` exceeds `max_sample_size` \n"
                   "downsampling `sample1`...")
            warn(msg)
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            msg = ("\n `sample2` exceeds `max_sample_size` \n"
                    "downsampling `sample2`...")
            warn(msg)
    
    rbins = convert_to_ndarray(rbins)
    rmax = np.max(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = ("\n Input `rbins` must be a monotonically increasing \n"
               "1-D array with at least two entries.")
        raise HalotoolsError(msg)
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "\n Input `period` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    #check for input parameter consistency
    if (period is not None):
        if (rmax >= np.min(period)/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                   "cannot be larger than Lbox/3 in any dimension. \n"
                   "If you need to count pairs on these length scales, \n"
                   "you should use a larger simulation.\n")
            raise HalotoolsError(msg)
    
    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\n `sample1` and `sample2` must have same dimension.')
        raise HalotoolsError(msg)
    
    if (randoms is None) & (PBCs is False):
        msg = ('\n If no PBCs are specified, randoms must be provided.')
        raise HalotoolsError(msg)
    
    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('\n `do_auto` and `do_cross` keywords must be of type boolean.')
        raise HalotoolsError(msg)
    
    if num_threads == 'max':
        num_threads = cpu_count()
    
    available_estimators = _list_estimators()
    if estimator not in available_estimators:
        msg = ("\n Input `estimator` must be one of the following: \n"
               "{0}".value(available_estimators))
        raise HalotoolsError(msg)
    
    return sample1, sample1_host_halo_id, rbins, sample2, sample2_host_halo_id,\
        randoms, period, do_auto, do_cross, num_threads, _sample1_is_sample2, PBCs


