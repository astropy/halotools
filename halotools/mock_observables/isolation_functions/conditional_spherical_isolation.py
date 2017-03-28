r"""
Module containing the `~halotools.mock_observables.conditional_spherical_isolation` function
used to apply a a variety of 3d isolation criteria to a set of points in a periodic box.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from functools import partial
import multiprocessing

from .spherical_isolation import _spherical_isolation_process_args
from .isolation_functions_helpers import _conditional_isolation_process_marks
from .engines import marked_spherical_isolation_engine

from ..pair_counters.rectangular_mesh import RectangularDoubleMesh
from ..pair_counters.mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices

__all__ = ('conditional_spherical_isolation', )

__author__ = ['Duncan Campbell', 'Andrew Hearin']

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero


def conditional_spherical_isolation(sample1, sample2, r_max,
        marks1=None, marks2=None, cond_func=0, period=None,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Determine whether a set of points, ``sample1``, is isolated, i.e. does not have a
    neighbor in ``sample2`` within an user specified spherical volume centered at each
    point in ``sample1``, where various additional conditions may be applied to judge
    whether a matching point is considered to be a neighbor.

    For example, `conditional_spherical_isolation` can be used to identify galaxies as
    isolated if no other galaxy with a greater stellar mass lies within 500 kpc.
    Different additional criteria can be built up from different combinations of
    input ``marks1``, ``marks2`` and ``cond_func``.

    See the Examples section for further details, and also
    :ref:`galaxy_catalog_intermediate_analysis_tutorial1` for a tutorial on usage
    with a mock galaxy catalog.

    Parameters
    ----------
    sample1 : array_like
        *Npts1 x 3* numpy array containing 3-D positions of points.

        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like
        *Npts2 x 3* numpy array containing 3-D positions of points.

    r_max : array_like
        radius of spheres to search for neighbors around galaxies in ``sample1``.
        If a single float is given, r_max is assumed to be the same for each galaxy in
        ``sample1``. You may optionally pass in an array of length *Npts1*, in which case
        each point in ``sample1`` will have its own individual neighbor-search radius.

        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    marks1 : array_like, optional
        *Npts1 x N_marks* array of marks.  The supplied marks array must have the
        appropriate shape for the chosen ``cond_func`` (see Notes for requirements).
        If this parameter is not specified, all marks will be set to unity.

    marks2 : array_like, optional
        *Npts2 x N_marks* array of marks.  The supplied marks array must have the
        appropriate shape for the chosen ``cond_func`` (see Notes for requirements).
        If this parameter is not specified, all marks will be set to unity.

    cond_func : int, optional
        Integer ID indicating which function should be used to apply an additional
        condition on whether a nearby point should be considered as a candidate neighbor.
        This allows, for example, stellar mass-dependent isolation criteria on a
        galaxy-by-galaxy basis.

        Default is 0 for an unconditioned calculation, in which case
        points will be considered neighbor candidates regardless of the
        value of their marks.
        See Notes for a list of options for the conditional functions.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

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
        Default choice is to use ``r_max``/10 in each dimension,
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
    different number of marks per point, N_marks.
    All conditional functions return a boolean and get passed
    two arrays per pair, *w1* and *w2*, each of length N_marks.
    You can pass in more than one piece of information about each point by choosing a
    the input ``marks`` arrays to be multi-dimensional of shape (N_points, N_marks).

    The available marking functions, ``cond_func``, and the associated integer
    ID numbers are:

    0. trivial (N_marks = 1)
        .. math::
            f(w_1,w_2) = True

    1. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\
                    False & : w_1[0] \leq w_2[0] \\
                \end{array}
                \right.

    2. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\
                    False & : w_1[0] \geq w_2[0] \\
                \end{array}
                \right.

    3. equality (N_marks = 1)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    True & : w_1[0] = w_2[0] \\
                    False & : w_1[0] \neq w_2[0] \\
                \end{array}
                \right.

    4. inequality (N_marks = 1)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    True & : w_1[0] \neq w_2[0] \\
                    False & : w_1[0] = w_2[0] \\
                \end{array}
                \right.

    5. tolerance greater than (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    True & : w_1[0] > (w_2[0]+w_1[1]) \\
                    False & : w_1[0] \leq (w_2[0]+w_1[1]) \\
                \end{array}
                \right.

    6. tolerance less than (N_marks = 2)
        .. math::
            f(w_1,w_2) =
                \left \{
                \begin{array}{ll}
                    True & : w_1[0] < (w_2[0]+w_1[1]) \\
                    False & : w_1[0] \geq (w_2[0]+w_1[1]) \\
                \end{array}
                \right.

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

    # Process the inputs with the helper function
    result = _spherical_isolation_process_args(sample1, sample2, r_max, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    r_max, max_r_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period

    search_xlength, search_ylength, search_zlength = max_r_max, max_r_max, max_r_max

    # Compute the estimates for the cell sizes
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

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    # Process the input marks and with the helper function
    marks1, marks2 = _conditional_isolation_process_marks(sample1, sample2, marks1, marks2, cond_func)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(marked_spherical_isolation_engine,
        double_mesh, x1in, y1in, z1in, x2in, y2in, z2in,
        marks1, marks2, cond_func, r_max)

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
