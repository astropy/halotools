# -*- coding: utf-8 -*-

"""
Helper functions used to process the arguments passed 
to functions in the void_stats module. 
"""
import numpy as np 
from ..utils.array_utils import convert_to_ndarray, array_is_monotonic
from ..custom_exceptions import *

__all__ = ('_void_prob_func_process_args', 
    '_underdensity_prob_func_process_args')
__author__ = ('Andrew Hearin', )

def _void_prob_func_process_args(sample1, rbins, 
    n_ran, random_sphere_centers, period, num_threads,
    approx_cell1_size, approx_cellran_size):
    """
    """
    sample1 = convert_to_ndarray(sample1)

    rbins = convert_to_ndarray(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        assert np.min(rbins) > 0
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = ("\n Input ``rbins`` must be a monotonically increasing \n"
               "1-D array with at least two entries. All entries must be strictly positive.")
        raise HalotoolsError(msg)

    if period is None:
        xmin, xmax = np.min(sample1), np.max(sample1)
        ymin, ymax = np.min(sample1), np.max(sample1)
        zmin, zmax = np.min(sample1), np.max(sample1)
    else:
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = ("\nInput ``period`` must either be a float or length-3 sequence")
            raise HalotoolsError(msg)
        xmin, xmax = 0., float(period[0])
        ymin, ymax = 0., float(period[1])
        zmin, zmax = 0., float(period[2])

    if (n_ran is None):
        if (random_sphere_centers is None):
            msg = ("You must pass either ``n_ran`` or ``random_sphere_centers``")
            raise HalotoolsError(msg)
        else:
            random_sphere_centers = convert_to_ndarray(random_sphere_centers)
            try:
                assert random_sphere_centers.shape[1] == 3
            except AssertionError:
                msg = ("Your input ``random_sphere_centers`` must have shape (Nspheres, 3)")
                raise HalotoolsError(msg)
        n_ran = float(random_sphere_centers.shape[0])
    else:
        if random_sphere_centers is not None:
            msg = ("If passing in ``random_sphere_centers``, do not also pass in ``n_ran``.")
            raise HalotoolsError(msg)
        else:
            xran = np.random.uniform(xmin, xmax, n_ran)
            yran = np.random.uniform(ymin, ymax, n_ran)
            zran = np.random.uniform(zmin, zmax, n_ran)
            random_sphere_centers = np.vstack([xran, yran, zran]).T

    return (sample1, rbins, n_ran, random_sphere_centers, 
        period, num_threads, approx_cell1_size, approx_cellran_size)

def _underdensity_prob_func_process_args(sample1, rbins, 
    n_ran, random_sphere_centers, period, 
    sample_volume, u, num_threads,
    approx_cell1_size, approx_cellran_size):
    """
    """
    sample1 = convert_to_ndarray(sample1)

    rbins = convert_to_ndarray(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        assert np.min(rbins) > 0
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = ("\n Input ``rbins`` must be a monotonically increasing \n"
               "1-D array with at least two entries. All entries must be strictly positive.")
        raise HalotoolsError(msg)

    if period is None:
        xmin, xmax = np.min(sample1), np.max(sample1)
        ymin, ymax = np.min(sample1), np.max(sample1)
        zmin, zmax = np.min(sample1), np.max(sample1)
        if sample_volume is None:
            msg = ("If period is None, you must pass in ``sample_volume``.")
            raise HalotoolsError(msg)
        else:
            sample_volume = float(sample_volume)
    else:
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = ("\nInput ``period`` must either be a float or length-3 sequence")
            raise HalotoolsError(msg)
        xmin, xmax = 0., float(period[0])
        ymin, ymax = 0., float(period[1])
        zmin, zmax = 0., float(period[2])
        if sample_volume is None:
            sample_volume = period.prod()
        else:
            msg = ("If period is not None, do not pass in sample_volume")
            raise HalotoolsError(msg)

    if (n_ran is None):
        if (random_sphere_centers is None):
            msg = ("You must pass either ``n_ran`` or ``random_sphere_centers``")
            raise HalotoolsError(msg)
        else:
            random_sphere_centers = convert_to_ndarray(random_sphere_centers)
            try:
                assert random_sphere_centers.shape[1] == 3
            except AssertionError:
                msg = ("Your input ``random_sphere_centers`` must have shape (Nspheres, 3)")
                raise HalotoolsError(msg)
        n_ran = float(random_sphere_centers.shape[0])
    else:
        if random_sphere_centers is not None:
            msg = ("If passing in ``random_sphere_centers``, do not also pass in ``n_ran``.")
            raise HalotoolsError(msg)
        else:
            xran = np.random.uniform(xmin, xmax, n_ran)
            yran = np.random.uniform(ymin, ymax, n_ran)
            zran = np.random.uniform(zmin, zmax, n_ran)
            random_sphere_centers = np.vstack([xran, yran, zran]).T

    u = float(u)

    return (sample1, rbins, n_ran, random_sphere_centers, period, 
        sample_volume, u, num_threads, approx_cell1_size, approx_cellran_size)









