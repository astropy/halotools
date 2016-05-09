"""
Module containing the `~halotools.mock_observables.spherical_isolation` function 
used to apply a simple 3d isolation criteria to a set of points in a periodic box.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from functools import partial 
import multiprocessing 

from ..pair_counters.rectangular_mesh import RectangularDoubleMesh
from ..pair_counters.cpairs import spherical_isolation_engine
from ..pair_counters.mesh_helpers import (
    _set_approximate_cell_sizes, _cell1_parallelization_indices, _enclose_in_box)

from ...utils.array_utils import convert_to_ndarray, custom_len

__all__ = ('spherical_isolation', )

__author__ = ['Andrew Hearin', 'Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero

def spherical_isolation(sample1, sample2, r_max, period=None,
    num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, is isolated, i.e. does not have a 
    neighbor in ``sample2`` within an user specified spherical volume centered at each 
    point in ``sample1``.

    See also :ref:`galaxy_catalog_analysis_tutorial10` for example usage on a 
    mock galaxy catalog. 
    
    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        
        See the :ref:`mock_obs_pos_formatting` documentation page, or the 
        Examples section below, for instructions on how to transform 
        your coordinate position arrays into the 
        format accepted by the ``sample1`` and ``sample2`` arguments.   
        
        Length units assumed to be in Mpc/h, here and throughout Halotools. 
        
    sample2 : array_like
        Npts2 x 3 numpy array containing 3-D positions of points.
        
    r_max : array_like
        radius of spheres to search for neighbors around galaxies in ``sample1``.
        If a single float is given, r_max is assumed to be the same for each galaxy in
        ``sample1``. Length units assumed to be in Mpc/h, here and throughout Halotools.
        
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 
        Length units assumed to be in Mpc/h, here and throughout Halotools. 
        
    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed 
        using the python ``multiprocessing`` module. Default is 1 for a purely serial 
        calculation, in which case a multiprocessing Pool object will 
        never be instantiated. A string 'max' may be used to indicate that 
        the pair counters should use all available cores on the machine.
        
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by how points 
        will be apportioned into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use Lbox/10 in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 
        
    approx_cell2_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for ``sample2``.  See comments for 
        ``approx_cell1_size`` for details. 
        
    Returns
    -------
    is_isolated : numpy.array
        array of booleans indicating if each point in `sample1` is isolated.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points ``sample1``.
    We will use the `~halotools.mock_observables.spherical_isolation` function to 
    determine which points in ``sample1`` are a distance ``r_max`` greater than all 
    other points in the sample.
    
    >>> Npts = 1000
    >>> Lbox = 250.0
    >>> period = Lbox
    
    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> sample1 = np.vstack((x,y,z)).T
    
    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array`
    convenience function for this same purpose, which provides additional wrapper 
    behavior around `numpy.vstack` such as placing points into redshift-space. 
    
    Now we will call `spherical_isolation` with ``sample2`` set to ``sample1``, 
    requiring that no other galaxy be located within 500 kpc/h in order for a point 
    to be considered isolated. Recall that Halotools assumes *h=1* and that all length-units 
    are in Mpc/h throughout the package. 
    
    >>> r_max = 0.5 # isolation cut of 500 kpc/h 
    >>> is_isolated = spherical_isolation(sample1, sample1, r_max, period=period)
    
    In the next example that follows, ``sample2`` will be a different set of points 
    from ``sample1``, so we will determine which points in ``sample1`` are located 
    greater than distance ``r_max`` away from all points in ``sample2``. 
    
    >>> x2 = np.random.uniform(0, Lbox, Npts)
    >>> y2 = np.random.uniform(0, Lbox, Npts)
    >>> z2 = np.random.uniform(0, Lbox, Npts)
    >>> sample2 = np.vstack([x2, y2, z2]).T
    >>> is_isolated = spherical_isolation(sample1, sample2, r_max, period=period)
    
    Notes
    -----
    There is one edge-case of all the isolation criteria functions worthy of special mention. 
    Suppose there exists a point *p* in ``sample1`` with the exact same spatial coordinates 
    as one or more points in ``sample2``. The matching point(s) in ``sample2`` will **not** 
    be considered neighbors of *p*. 
    
    """
    
    ### Process the inputs with the helper function
    result = _spherical_isolation_process_args(sample1, sample2, r_max, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    r_max, max_r_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 
    
    search_xlength, search_ylength, search_zlength = max_r_max, max_r_max, max_r_max 
    
    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
    
    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)
    
    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(spherical_isolation_engine, 
        double_mesh, sample1[:,0], sample1[:,1], sample1[:,2], 
        sample2[:,0], sample2[:,1], sample2[:,2], r_max)
    
    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)
    
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        counts = np.sum(np.array(result), axis=0)
        pool.close()
    else:
        counts = engine(cell1_tuples[0])
    
    is_isolated = np.array(counts, dtype=bool)
    
    return is_isolated




def _spherical_isolation_process_args(data1, data2, r_max, period, 
    num_threads, approx_cell1_size, approx_cell2_size):
    """
    private function to process the arguents for spherical isolation functions
    """
    if num_threads is not 1:
        if num_threads=='max':
            num_threads = multiprocessing.cpu_count()
        if not isinstance(num_threads,int):
            msg = "Input ``num_threads`` argument must be an integer or the string 'max'"
            raise ValueError(msg)
    
    # Passively enforce that we are working with ndarrays
    x1 = data1[:,0]
    y1 = data1[:,1]
    z1 = data1[:,2]
    x2 = data2[:,0]
    y2 = data2[:,1]
    z2 = data2[:,2]
    
    N1 = len(x1)
    
    r_max = convert_to_ndarray(r_max).astype(float)
    if len(r_max) == 1:
        r_max = np.array([r_max[0]]*N1)
    try:
        assert np.all(r_max < np.inf)
        assert np.all(r_max > 0)
    except AssertionError:
        msg = "Input ``r_max`` must be an array of bounded positive numbers."
        raise ValueError(msg)
    
    max_r_max = np.amax(r_max)
    
    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2, 
                min_size=[max_r_max*3.0,max_r_max*3.0,max_r_max*3.0]))
    else:
        PBCs = True
        period = convert_to_ndarray(period).astype(float)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise ValueError(msg)

    try:
        assert max_r_max < period[0]/3.
        assert max_r_max < period[1]/3.
        assert max_r_max < period[2]/3.
    except AssertionError:
        msg = ("Input ``max(r_max)`` must be less than input period/3 in all dimensions.")
        raise ValueError(msg)

    if approx_cell1_size is None:
        approx_cell1_size = [max_r_max, max_r_max, max_r_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:    
        approx_cell2_size = [max_r_max, max_r_max, max_r_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]
        
    return (x1, y1, z1, x2, y2, z2, 
        r_max, max_r_max, period, num_threads, PBCs, 
        approx_cell1_size, approx_cell2_size)






