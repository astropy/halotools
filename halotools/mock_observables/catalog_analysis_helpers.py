# -*- coding: utf-8 -*-

""" Common functions used when analyzing 
catalogs of galaxies/halos.

"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

from copy import deepcopy 

import numpy as np
from scipy.stats import binned_statistic

from ..custom_exceptions import HalotoolsError

__all__ = ('mean_y_vs_x', )

def mean_y_vs_x(x, y, error_estimator = 'error_on_mean', **kwargs):
    """
    Estimate the mean value of the property *y* as a function of *x* 
    for an input sample of galaxies/halos, 
    optionally returning an error estimate. 

    The `mean_y_vs_x` function is just a convenience wrapper 
    around `scipy.stats.binned_statistic` and `np.histogram`. 

    Parameters 
    -----------
    x : array_like 
        Array storing values of the independent variable of the sample. 

    y : array_like 
        Array storing values of the dependent variable of the sample. 

    bins : array_like, optional 
        Bins of the input *x*. 
        Defaults are set by `scipy.stats.binned_statistic`.  

    error_estimator : string, optional 
        If set to ``error_on_mean``, function will also return an array storing  
        :math:`\\sigma_{y}/\\sqrt{N}`, where :math:`\\sigma_{y}` is the 
        standard deviation of *y* in the bin 
        and :math:`\\sqrt{N}` is the counts in each bin. 

        If set to ``variance``, function will also return an array storing  
        :math:`\\sigma_{y}`. 

        Default is ``error_on_mean``

    Returns 
    ----------
    bin_midpoints : array_like 
        Midpoints of the *x*-bins. 

    mean : array_like 
        Mean of *y* estimated in bins 

    err : array_like 
        Error on *y* estimated in bins

    Examples 
    ---------
    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> halos = halocat.halo_table 
    >>> halo_mass, mean_spin, err = mean_y_vs_x(halos['halo_mvir'], halos['halo_spin'])

    """
    try:
        assert error_estimator in ('error_on_mean', 'variance')
    except AssertionError:
        msg = ("\nInput ``error_estimator`` must be either "
            "``error_on_mean`` or ``variance``\n")
        raise HalotoolsError(msg)

    modified_kwargs = {key:kwargs[key] for key in kwargs if key != 'error_estimator'}
    result = binned_statistic(x, y, statistic='mean', **modified_kwargs)
    mean, bin_edges, binnumber = result
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1])/2.

    modified_kwargs['bins'] = bin_edges

    result = binned_statistic(x, y, statistic=np.std, **modified_kwargs)
    variance, _, _ = result

    if error_estimator == 'variance':
        err = variance
    else:
        counts = np.histogram(x, bins = bin_edges)
        err = variance/np.sqrt(counts[0])

    return bin_midpoints, mean, err








