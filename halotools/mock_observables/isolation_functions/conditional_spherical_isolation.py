"""
Module containing the `~halotools.mock_observables.conditional_spherical_isolation` function 
used to apply a a variety of 3d isolation criteria to a set of points in a periodic box.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from functools import partial 
import multiprocessing 

from .spherical_isolation  import _spherical_isolation_process_args

from ..pair_counters.rectangular_mesh import RectangularDoubleMesh
from ..pair_counters.marked_cpairs import (
    marked_spherical_isolation_engine, marked_cylindrical_isolation_engine)
from ..pair_counters.mesh_helpers import (
    _set_approximate_cell_sizes, _cell1_parallelization_indices, _enclose_in_box)

from ...utils.array_utils import convert_to_ndarray, custom_len
from ...custom_exceptions import HalotoolsError

__all__ = ('conditional_spherical_isolation', )

__author__ = ['Duncan Campbell', 'Andrew Hearin']

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero

def conditional_spherical_isolation(sample1, sample2, r_max,
    marks1, marks2, cond_func, period=None,
    num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, is isolated, i.e. does not have a 
    neighbor in ``sample2`` within an user specified spherical volume centered at each 
    point in ``sample1``, where various additional conditions may be applied to judge 
    whether a matching point is considered to be a neighbor. 
    
    For example, `conditional_spherical_isolation` can be used to identify galaxies as 
    isolated if no other galaxy with a greater stellar mass lies within 500 kpc. 
    Different additional criteria can be built up from different combinations of 
    input ``marks1``, ``marks2`` and ``cond_func``.
    
    See the Examples section for further details.
    
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
        If a single float is given, ``r_max`` is assumed to be the same for each galaxy in
        ``sample1``. Length units assumed to be in Mpc/h, here and throughout Halotools.
    
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
        Length units assumed to be in Mpc/h, here and throughout Halotools. 
    
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
        Analogous to ``approx_cell1_size``, but for ``sample2``.  See comments for 
        ``approx_cell1_size`` for details. 
        
    Returns
    -------
    is_isolated : numpy.array
        array of booleans indicating if each point in `sample1` is isolated.
    
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
    r_max, max_r_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    search_xlength, search_ylength, search_zlength = max_r_max, max_r_max, max_r_max 

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

def _conditional_isolation_process_weights(sample1, sample2, weights1, weights2, cond_func):
    """
    private function to process the arguents for conditional isolation functions
    """
    
    correct_num_weights = _func_signature_int_from_cond_func(cond_func)
    npts_sample1 = np.shape(sample1)[0]
    npts_sample2 = np.shape(sample2)[0]
    correct_shape1 = (npts_sample1, correct_num_weights)
    correct_shape2 = (npts_sample2, correct_num_weights)
    
    ### Process the input weights1
    _converted_to_2d_from_1d = False
    # First convert weights1 into a 2-d ndarray
    if weights1 is None:
        weights1 = np.ones((npts_sample1, 1), dtype = np.float64)
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
                   "points in `sample1` = %i, while the number of points \n"
                   "in your input 1-D `weights1` array = %i")
            raise HalotoolsError(msg % (npts_sample1, npts_weights1))
        else:
            msg = ("\n You passed in a 2-D array for `weights1` that \n"
                   "does not have a consistent shape with `sample1`. \n"
                   "`sample1` has length %i. The input value of `cond_func` = %i \n"
                   "For this value of `cond_func`, there should be %i weights \n"
                   "per point. The shape of your input `weights1` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_sample1, cond_func, correct_num_weights, npts_weights1, num_weights1))
    
    ### Process the input weights2
    _converted_to_2d_from_1d = False
    # Now convert weights2 into a 2-d ndarray
    if weights2 is None:
        weights2 = np.ones((npts_sample2, 1), dtype = np.float64)
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
                   "points in `sample2` = %i, while the number of points \n"
                   "in your input 1-D `weights2` array = %i")
            raise HalotoolsError(msg % (npts_sample2, npts_weights2))
        else:
            msg = ("\n You passed in a 2-D array for `weights2` that \n"
                   "does not have a consistent shape with `sample2`. \n"
                   "`sample2` has length %i. The input value of `cond_func` = %i \n"
                   "For this value of `cond_func`, there should be %i weights \n"
                   "per point. The shape of your input `weights2` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_sample2, cond_func, correct_num_weights, npts_weights2, num_weights2))
    
    return weights1, weights2

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


