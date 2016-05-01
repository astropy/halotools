"""
Module containing functions used to determine whether 
a set of points are isolated according to various criteria.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from functools import partial 
import multiprocessing 

from .pair_counters.rectangular_mesh import RectangularDoubleMesh
from .pair_counters.pair_counting_engines import spherical_isolation_engine, cylindrical_isolation_engine
from .pair_counters.mesh_helpers import (
    _set_approximate_cell_sizes, _cell1_parallelization_indices, _enclose_in_box)

from .pair_counters.marked_pair_counting_engines import (
    marked_spherical_isolation_engine, marked_cylindrical_isolation_engine)

from ..utils.array_utils import convert_to_ndarray, custom_len
from ..custom_exceptions import HalotoolsError

__all__ = ('spherical_isolation', 'cylindrical_isolation',
    'conditional_spherical_isolation', 'conditional_cylindrical_isolation')

__author__ = ['Duncan Campbell', 'Andrew Hearin']

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def spherical_isolation(sample1, sample2, r_max, period=None,
    num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    an input spherical volume centered at each point in ``sample1``.

    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.

        See the :ref:`mock_obs_pos_formatting` documentation page, or the 
        Examples section below, for instructions on how to transform 
        your coordinate position arrays into the 
        format accepted by the ``sample1`` and ``sample2`` arguments.   
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    r_max : float
        size of sphere to search for neighbors
    
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 
    
    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed 
        using the python ``multiprocessing`` module. Default is 1 for a purely 
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
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for 
        ``approx_cell1_size`` for details. 
    
    Returns
    -------
    is_isolated : numpy.array
        array of booleans indicating if the point is isolated.
        
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points ``sample1``. 
    We will use the `~halotools.mock_observables.spherical_isolation` function to determine 
    which points in ``sample1`` are a distance ``r_max`` greater than all other points in the sample.
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = Lbox
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> sample1 = np.vstack((x,y,z)).T

    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array` 
    convenience function for this same purpose, which provides additional wrapper 
    behavior around `numpy.vstack` such as placing points into redshift-space. 

    Now we will call `spherical_isolation` with ``sample2`` set to ``sample1``:

    >>> r_max = 0.05
    >>> is_isolated = spherical_isolation(sample1, sample1, r_max, period=period)

    In the next example that follows, ``sample2`` will be a different set of points 
    from ``sample1``, so we will determine which points in ``sample1`` are located 
    greater than distance ``r_max`` away from all points in ``sample2``. 

    >>> sample2 = np.random.random((Npts, 3))
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
    r_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 
    
    search_xlength, search_ylength, search_zlength = r_max, r_max, r_max 
    
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


def cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=None,
    num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    an input cylindrical volume centered at each point in ``sample1``.
        
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.

        See the :ref:`mock_obs_pos_formatting` documentation page, or the 
        Examples section below, for instructions on how to transform 
        your coordinate position arrays into the 
        format accepted by the ``sample1`` and ``sample2`` arguments.   
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    rp_max : float
        radius of the cylinder to seach for neighbors
    
    pi_max : float
        half-length of the cylinder to seach for neighbors
    
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 
    
    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed 
        using the python ``multiprocessing`` module. Default is 1 for a purely 
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
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for 
        ``approx_cell1_size`` for details. 
        
    Returns
    -------
    is_isolated : numpy.array
        array of booleans indicating if the point is isolated.
        
    Examples
    --------
    First we create a randomly distributed set of points ``sample1``, together with 
    random z-velocities for those points. We will then place ``sample1`` into redshift-space 
    using the `~halotools.mock_observables.return_xyz_formatted_array` function. 
    We will use the `~halotools.mock_observables.cylindrical_isolation` function to determine 
    which points in ``sample1`` have zero neighbors inside a cylinder of radius 
    ``rp_max`` and half-length ``pi_max``. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = Lbox
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    >>> vz = np.random.normal(loc = 0, scale = 100, size = Npts)
    
    We place our points into redshift-space, formatting the result into the 
    appropriately shaped array used throughout the `~halotools.mock_observables` sub-package:
    
    >>> from halotools.mock_observables import return_xyz_formatted_array
    >>> sample1 = return_xyz_formatted_array(x, y, z, period = Lbox, velocity = vz, velocity_distortion_dimension='z')
    
    Now we will call `cylindrical_isolation` with ``sample2`` set to ``sample1``:

    >>> rp_max = 0.05
    >>> pi_max = 0.1
    >>> is_isolated = cylindrical_isolation(sample1, sample1, rp_max, pi_max, period=period)

    In the next example that follows, ``sample2`` will be a different set of points 
    from ``sample1``, so we will determine which points in ``sample1`` 
    have a neighbor in ``sample2`` located inside a cylinder of radius ``rp_max`` 
    and half-length ``pi_max``. 

    >>> x2 = np.random.random(Npts)
    >>> y2 = np.random.random(Npts)
    >>> z2 = np.random.random(Npts)
    >>> vz2 = np.random.normal(loc = 0, scale = 100, size = Npts)
    >>> sample2 = return_xyz_formatted_array(x2, y2, z2, period = Lbox, velocity = vz2, velocity_distortion_dimension='z')

    >>> is_isolated = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=period)

    Notes
    -----
    There is one edge-case of all the isolation criteria functions worthy of special mention. 
    Suppose there exists a point *p* in ``sample1`` with the exact same spatial coordinates 
    as one or more points in ``sample2``. The matching point(s) in ``sample2`` will **not** 
    be considered neighbors of *p*. 
    """
    
    ### Process the inputs with the helper function
    result = _cylindrical_isolation_process_args(sample1, sample2, rp_max, pi_max, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rp_max, pi_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 
    
    search_xlength, search_ylength, search_zlength = rp_max, rp_max, pi_max 
    
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
    engine = partial(cylindrical_isolation_engine, 
        double_mesh, sample1[:,0], sample1[:,1], sample1[:,2], 
        sample2[:,0], sample2[:,1], sample2[:,2], rp_max, pi_max)
    
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


def conditional_spherical_isolation(sample1, sample2, r_max,
    marks1, marks2, cond_func, period=None,
    num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    an input spherical volume centered at each point in ``sample1``, 
    where various additional conditions may be applied to judge whether a matching point 
    is considered to be a neighbor. For example, 
    `conditional_spherical_isolation` can be used to identify galaxies as isolated 
    if no other galaxy with a greater stellar mass lies within 500 kpc. 
    Different additional criteria can be built up from different 
    combinations of input ``marks`` and ``cond_func``. 
    See the Examples section for further details.  
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the 
        Examples section below, for instructions on how to transform 
        your coordinate position arrays into the 
        format accepted by the ``sample1`` and ``sample2`` arguments.   
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    r_max : float
        size of sphere to search for neighbors
    
    marks1 : array_like
        len(sample1) x N_marks array of marks.  The supplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    marks2 : array_like
        len(sample2) x N_marks array of marks.  The supplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    cond_func : int
        Integer ID indicating which conditional function should be used.  See notes for a 
        list of available conditional functions.
        
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 
    
    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed 
        using the python ``multiprocessing`` module. Default is 1 for a purely 
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
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for 
        ``approx_cell1_size`` for details. 
        
    Returns
    -------
    is_isolated : numpy.array
        array of booleans indicating if the point is isolated.
    
    Notes
    -----
    The `~halotools.mock_observables.conditional_spherical_isolation` function only differs 
    from the `~halotools.mock_observables.spherical_isolation` function in the treatment of 
    the input marks. In order for a point *p2* in ``sample2`` with mark :math:`w_{2}` 
    to be considered a neighbor of a point *p1* in ``sample1`` with mark :math:`w_{1}`, 
    two following conditions must be met:

    #. *p2* must lie within a distance ``r_max`` of *p1*, and 

    #. the input conditional marking function :math:`f(w_{1}, w_{2})` must return *True*.  

    There are multiple conditional functions available.  In general, each requires a 
    different number of marks per point, N_marks.  The conditional function gets passed 
    two arrays per pair, *w1* and *w2*, of length N_marks and return a boolean.  
    You can pass in more than one piece of information about each point by choosing a 
    the input ``marks`` arrays to be multi-dimensional of shape (N_points, N_marks). 
        
    The available marking functions, ``cond_func`` and the associated integer 
    ID numbers are:
    
    #. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\\\
                    False & : w_1[0] \\leq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\\\
                    False & : w_1[0] \\geq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. equality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] = w_2[0] \\\\
                    False & : w_1[0] \\neq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. inequality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] \\neq w_2[0] \\\\
                    False & : w_1[0] = w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. tolerance greater than (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\leq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    #. tolerance less than (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\geq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    Examples
    --------
    In this first example, we will show how to calculate the following notion of 
    galaxy isolation. A galaxy is isolated if there are zero other *more massive* 
    galaxies within 5 Mpc. 

    First we create a random distribution of points inside the box: 

    >>> Npts = 1000
    >>> Lbox = 250.
    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> sample1 = np.vstack((x,y,z)).T
    
    Now we will choose random stellar masses for our galaxies:
    
    >>> stellar_mass = np.random.uniform(1e10, 1e12, Npts)

    Since we are interested in whether a point in ``sample1`` is isolated from other points 
    in ``sample1``, we set ``sample2`` to ``sample1`` and both ``marks1`` and ``marks2`` 
    equal to ``stellar_mass``. 

    >>> sample2 = sample1
    >>> marks1 = stellar_mass
    >>> marks2 = stellar_mass

    Referring to the Notes above for the definitions of the conditional marking functions, 
    we see that for this particular isolation criteria the appropriate ``cond_func`` is 2. 
    The reason is that this function only evaluates to *True* for those points in ``sample2`` 
    that are more massive than the ``sample1`` point under consideration. Thus the only 
    relevant points to consider as candidate neighbors are the more massive ones; all other 
    ``sample2`` points will be disregarded irrespective of their distance from the 
    ``sample1`` point under consideration.
    
    >>> r_max = 5.0
    >>> cond_func = 2

    >>> is_isolated = conditional_spherical_isolation(sample1, sample2, r_max, marks1, marks2, cond_func, period=Lbox)

    """
    ### Process the inputs with the helper function
    result = _spherical_isolation_process_args(sample1, sample2, r_max, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    r_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    search_xlength, search_ylength, search_zlength = r_max, r_max, r_max 

    # Process the input weights and with the helper function
    marks1, marks2 = _conditional_isolation_process_weights(sample1, sample2, marks1, marks2, cond_func)

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
    engine = partial(marked_spherical_isolation_engine, 
        double_mesh, sample1[:,0], sample1[:,1], sample1[:,2], 
        sample2[:,0], sample2[:,1], sample2[:,2], marks1, marks2, cond_func, r_max)

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


def conditional_cylindrical_isolation(sample1, sample2, rp_max, pi_max,
                          marks1, marks2, cond_func, period=None, num_threads=1,
                          approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    an input cylindrical volume centered at each point in ``sample1``, 
    where various additional conditions may be applied to judge whether a matching point 
    is considered to be a neighbor. For example, 
    `conditional_cylindrical_isolation` can be used to identify galaxies as isolated 
    if no other galaxy with a greater stellar mass lies within 500 kpc. 
    Different additional criteria can be built up from different 
    combinations of input ``marks`` and ``cond_func``. 
    See the Examples section for further details.  
    
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.

        See the :ref:`mock_obs_pos_formatting` documentation page, or the 
        Examples section below, for instructions on how to transform 
        your coordinate position arrays into the 
        format accepted by the ``sample1`` and ``sample2`` arguments.   
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    rp_max : float
        radius of the cylinder to seach for neighbors
    
    pi_max : float
        half the length of the cylinder to seach for neighbors
    
    marks1 : array_like
        len(sample1) x N_marks array of marks.  The supplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    marks2 : array_like
        len(sample2) x N_marks array of marks.  The supplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    cond_func : int
        Integer ID indicating which conditional function should be used.  See notes for a 
        list of available conditional functions.
    
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 
    
    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed 
        using the python ``multiprocessing`` module. Default is 1 for a purely 
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
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for 
        ``approx_cell1_size`` for details.
    
    Returns
    -------
    is_isolated : numpy.array
        array of booleans indicating if the point is isolated.
    
    Notes
    -----
    The `~halotools.mock_observables.conditional_cylindrical_isolation` function only differs 
    from the `~halotools.mock_observables.cylindrical_isolation` function in the treatment of 
    the input marks. In order for a point *p2* in ``sample2`` with mark :math:`w_{2}` 
    to be considered a neighbor of a point *p1* in ``sample1`` with mark :math:`w_{1}`, 
    two following conditions must be met:

    #. *p2* must lie within an xy-distance ``rp_max`` and a z-distance ``pi_max`` of *p1*, and

    #. the input conditional marking function :math:`f(w_{1}, w_{2})` must return *True*.  

    There are multiple conditional functions available.  In general, each requires a 
    different number of marks per point, N_marks.  The conditional function gets passed 
    two arrays per pair, w1 and w2, of length N_marks and return a float.  
    You can pass in more than one piece of information about each point by choosing a 
    the input ``marks`` arrays to be multi-dimensional of shape (N_points, N_marks). 
        
    The available marking functions, ``cond_func`` and the associated integer 
    ID numbers are:
    
    #. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\\\
                    False & : w_1[0] \\leq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\\\
                    False & : w_1[0] \\geq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. equality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] = w_2[0] \\\\
                    False & : w_1[0] \\neq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. inequality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] \\neq w_2[0] \\\\
                    False & : w_1[0] = w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. tolerance greater than (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\leq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    #. tolerance less than (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\geq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    
    Examples
    --------
    In this first example, we will show how to calculate the following notion of 
    galaxy isolation. A galaxy is isolated if there are zero other *more massive* 
    galaxies within a projected distance of 750 kpc and a z-distance of 500 km/s. 
    
    First we create a random distribution of points inside the box, and also random z-velocities. 

    >>> Npts = 1000
    >>> Lbox = 250.
    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)
    >>> vz = np.random.normal(loc = 0, scale = 100, size = Npts)
    
    We place our points into redshift-space, formatting the result into the 
    appropriately shaped array used throughout the `~halotools.mock_observables` sub-package:
    
    >>> from halotools.mock_observables import return_xyz_formatted_array
    >>> sample1 = return_xyz_formatted_array(x, y, z, period = Lbox, velocity = vz, velocity_distortion_dimension='z')

    Now we will choose random stellar masses for our galaxies:
    
    >>> stellar_mass = np.random.uniform(1e10, 1e12, Npts)

    Since we are interested in whether a point in ``sample1`` is isolated from other points 
    in ``sample1``, we set ``sample2`` to ``sample1`` and both ``marks1`` and ``marks2`` 
    equal to ``stellar_mass``. 

    >>> sample2 = sample1
    >>> marks1 = stellar_mass
    >>> marks2 = stellar_mass

    All units in Halotools assume *h=1*, with lengths always in Mpc/h, so we have:

    >>> rp_max = 0.75

    Since *h=1* implies :math:`H_{0} = 100`km/s/Mpc, our 500 km/s velocity criteria 
    gets transformed into a z-dimension length criteria as:

    >>> H0 = 100.0
    >>> pi_max = 500./H0
    
    Referring to the Notes above for the definitions of the conditional marking functions, 
    we see that for this particular isolation criteria the appropriate ``cond_func`` is 2. 
    The reason is that this function only evaluates to *True* for those points in ``sample2`` 
    that are more massive than the ``sample1`` point under consideration. Thus the only 
    relevant points to consider as candidate neighbors are the more massive ones; all other 
    ``sample2`` points will be disregarded irrespective of their distance from the 
    ``sample1`` point under consideration.

    >>> cond_func = 2
    >>> is_isolated = conditional_cylindrical_isolation(sample1, sample2, rp_max, pi_max, marks1, marks2, cond_func, period=Lbox)
    """
    
    ### Process the inputs with the helper function
    result = _cylindrical_isolation_process_args(sample1, sample2, rp_max, pi_max, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rp_max, pi_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 
    
    search_xlength, search_ylength, search_zlength = rp_max, rp_max, pi_max 
    
    # Process the input weights and with the helper function
    marks1, marks2 = _conditional_isolation_process_weights(sample1, sample2, marks1, marks2, cond_func)
    
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
    engine = partial(marked_cylindrical_isolation_engine, 
        double_mesh, sample1[:,0], sample1[:,1], sample1[:,2], 
        sample2[:,0], sample2[:,1], sample2[:,2], marks1, marks2, cond_func, rp_max, pi_max)
    
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


def _cylindrical_isolation_process_args(data1, data2, rp_max, pi_max, period, 
    num_threads, approx_cell1_size, approx_cell2_size):
    """
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
        
    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2, 
                min_size=[rp_max*3.0,rp_max*3.0,rp_max*3.0]))
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
        assert rp_max < period[0]/3.
        assert rp_max < period[1]/3.
        assert pi_max < period[2]/3.
    except AssertionError:
        msg = ("Input ``rp_max`` and ``pi_max`` must both be less than "
            "input period in the first two and third dimensions respectively.")
        raise ValueError(msg)
    
    if approx_cell1_size is None:
        approx_cell1_size = [rp_max, rp_max, pi_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:    
        approx_cell2_size = [rp_max, rp_max, pi_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]
    
    return (x1, y1, z1, x2, y2, z2, 
        rp_max, pi_max, period, num_threads, PBCs, 
        approx_cell1_size, approx_cell2_size)


def _conditional_isolation_process_weights(data1, data2, weights1, weights2, cond_func):
    """
    """
    
    correct_num_weights = _func_signature_int_from_cond_func(cond_func)
    npts_data1 = np.shape(data1)[0]
    npts_data2 = np.shape(data2)[0]
    correct_shape1 = (npts_data1, correct_num_weights)
    correct_shape2 = (npts_data2, correct_num_weights)
    
    ### Process the input weights1
    _converted_to_2d_from_1d = False
    # First convert weights1 into a 2-d ndarray
    if weights1 is None:
        weights1 = np.ones((npts_data1, 1), dtype = np.float64)
    else:
        weights1 = convert_to_ndarray(weights1)
        weights1 = weights1.astype("float64")
        if weights1.ndim == 1:
            _converted_to_2d_from_1d = True
            npts1 = len(weights1)
            weights1 = weights1.reshape((npts1, 1))
        elif weights1.ndim == 2:
            pass
        else:
            ndim1 = weights1.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights1`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim1)
    
    npts_weights1 = np.shape(weights1)[0]
    num_weights1 = np.shape(weights1)[1]
    # At this point, weights1 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    
    if np.shape(weights1) != correct_shape1:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights1` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data1` = %i, while the number of points \n"
                   "in your input 1-D `weights1` array = %i")
            raise HalotoolsError(msg % (npts_data1, npts_weights1))
        else:
            msg = ("\n You passed in a 2-D array for `weights1` that \n"
                   "does not have a consistent shape with `data1`. \n"
                   "`data1` has length %i. The input value of `cond_func` = %i \n"
                   "For this value of `cond_func`, there should be %i weights \n"
                   "per point. The shape of your input `weights1` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data1, cond_func, correct_num_weights, npts_weights1, num_weights1))
    
    ### Process the input weights2
    _converted_to_2d_from_1d = False
    # Now convert weights2 into a 2-d ndarray
    if weights2 is None:
        weights2 = np.ones((npts_data2, 1), dtype = np.float64)
    else:
        weights2 = convert_to_ndarray(weights2)
        weights2 = weights2.astype("float64")
        if weights2.ndim == 1:
            _converted_to_2d_from_1d = True
            npts2 = len(weights2)
            weights2 = weights2.reshape((npts2, 1))
        elif weights2.ndim == 2:
            pass
        else:
            ndim2 = weights2.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights2`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim2)
    
    npts_weights2 = np.shape(weights2)[0]
    num_weights2 = np.shape(weights2)[1]
    # At this point, weights2 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights2) != correct_shape2:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights2` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data2` = %i, while the number of points \n"
                   "in your input 1-D `weights2` array = %i")
            raise HalotoolsError(msg % (npts_data2, npts_weights2))
        else:
            msg = ("\n You passed in a 2-D array for `weights2` that \n"
                   "does not have a consistent shape with `data2`. \n"
                   "`data2` has length %i. The input value of `cond_func` = %i \n"
                   "For this value of `cond_func`, there should be %i weights \n"
                   "per point. The shape of your input `weights2` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data2, cond_func, correct_num_weights, npts_weights2, num_weights2))
    
    return weights1, weights2


def _spherical_isolation_process_args(data1, data2, r_max, period, 
    num_threads, approx_cell1_size, approx_cell2_size):
    """
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
        
    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2, 
                min_size=[r_max*3.0,r_max*3.0,r_max*3.0]))
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
        assert r_max < period[0]/3.
        assert r_max < period[1]/3.
        assert r_max < period[2]/3.
    except AssertionError:
        msg = ("Input ``r_max`` must be less than input period/3 in all dimensions.")
        raise ValueError(msg)

    if approx_cell1_size is None:
        approx_cell1_size = [r_max, r_max, r_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:    
        approx_cell2_size = [r_max, r_max, r_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]
        
    return (x1, y1, z1, x2, y2, z2, 
        r_max, period, num_threads, PBCs, 
        approx_cell1_size, approx_cell2_size)


def _func_signature_int_from_cond_func(cond_func):
    """
    Return the function signature available weighting functions. 
    """
    
    if type(cond_func) != int:
        msg = "\n cond_func parameter must be an integer ID of a weighting function."
        raise ValueError(msg)
    
    if cond_func == 1:
        return 1
    elif cond_func == 2:
        return 1
    elif cond_func == 3:
        return 1
    elif cond_func == 4:
        return 1
    elif cond_func == 5:
        return 2
    elif cond_func == 6:
        return 2
    else:
        msg = ("The value ``cond_func`` = %i is not recognized")
        raise HalotoolsError(msg % cond_func)





