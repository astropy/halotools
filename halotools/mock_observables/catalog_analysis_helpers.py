# -*- coding: utf-8 -*-

""" Common functions used when analyzing 
catalogs of galaxies/halos.

"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

from copy import deepcopy 

import numpy as np
from scipy.stats import binned_statistic

from ..empirical_models import enforce_periodicity_of_box

from ..custom_exceptions import HalotoolsError

__all__ = ('mean_y_vs_x', 'return_xyz_formatted_array')
__author__ = ['Andrew Hearin']


def mean_y_vs_x(x, y, error_estimator = 'error_on_mean', **kwargs):
    """
    Estimate the mean value of the property *y* as a function of *x* 
    for an input sample of galaxies/halos, 
    optionally returning an error estimate. 

    The `mean_y_vs_x` function is just a convenience wrapper 
    around `scipy.stats.binned_statistic` and `np.histogram`. 

    See also :ref:`galaxy_catalog_analysis_tutorial1`. 

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

    See also 
    ---------
    :ref:`galaxy_catalog_analysis_tutorial1`
    
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

def return_xyz_formatted_array(x, y, z, period=np.inf, **kwargs):
    """ Returns a Numpy array of shape *(Npts, 3)* storing the 
    xyz-positions in the format used throughout
    the `~halotools.mock_observables` package. 

    See :ref:`mock_obs_pos_formatting` for a tutorial. 

    Parameters 
    -----------
    x, y, z : sequence of arrays 

    velocity : array, optional 
        Array used to apply peculiar velocity distortions, e.g.,  
        :math:`z_{\\rm dist} = z + v/H_{0}`. 
        Since Halotools workes exclusively in h=1 units, 
        in the above formula :math:`H_{0} = 100 km/s/Mpc`.

        If ``velocity`` argument is passed, 
        ``velocity_distortion_dimension`` must also be passed. 

    velocity_distortion_dimension : string, optional 
        If set to ``'x'``, ``'y'`` or ``'z'``, 
        the requested dimension in the returned ``pos`` array 
        will be distorted due to peculiar motion. 
        For example, if ``velocity_distortion_dimension`` is ``z``, 
        then ``pos`` can be treated as physically observed 
        galaxy positions under the distant-observer approximation. 
        Default is no distortions. 

    mask : array_like, optional 
        Boolean mask that can be used to select the positions 
        of a subcollection of the galaxies stored in the ``galaxy_table``. 

    period : float, optional 
        Length of the periodic box. Default is np.inf. 

        If period is not np.inf, then after applying peculiar velocity distortions 
        the new coordinates will be remapped into the periodic box. 

    Returns 
    --------
    pos : array_like 
        Numpy array with shape *(Npts, 3)*. 
    """
    posdict = {'x': np.copy(x), 'y': np.copy(y), 'z': np.copy(z)}

    a = 'velocity_distortion_dimension' in kwargs.keys()
    b = 'velocity' in kwargs.keys()
    if bool(a+b)==True:
        if bool(a*b)==False:
            msg = ("You must either both or none of the following keyword arguments: "
                "``velocity_distortion_dimension`` and ``velocity``\n")
            raise KeyError(msg)
        else:
            vel_dist_dim = kwargs['velocity_distortion_dimension']
            velocity = np.copy(kwargs['velocity'])
            apply_distortion = True
    else:
        apply_distortion = False

    if apply_distortion is True:
        try:
            assert vel_dist_dim in ('x', 'y', 'z')
            posdict[vel_dist_dim] = np.copy(posdict[vel_dist_dim]) + np.copy(velocity/100.)
            if period != np.inf:
                posdict[vel_dist_dim] = enforce_periodicity_of_box(
                    posdict[vel_dist_dim], period)
        except AssertionError:
            msg = ("\nInput ``velocity_distortion_dimension`` must be either \n"
                "``'x'``, ``'y'`` or ``'z'``.")
            raise KeyError(msg)

    xout, yout, zout = np.copy(posdict['x']), np.copy(posdict['y']), np.copy(posdict['z'])
    pos = np.vstack([xout, yout, zout]).T

    # Apply a mask, if applicable
    try:
        mask = kwargs['mask']
        return pos[mask]
    except KeyError:
        return pos





