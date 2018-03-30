"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing
from functools import partial

from .engines import inertia_tensor_per_object_engine

from ..mock_observables_helpers import get_num_threads, get_period, enforce_sample_respects_pbcs
from ..pair_counters.mesh_helpers import (_set_approximate_cell_sizes,
    _cell1_parallelization_indices, _enclose_in_box, _enforce_maximum_search_length)
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh


__all__ = ('inertia_tensor_per_object', )


def inertia_tensor_per_object(sample1, sample2, smoothing_scale, weights2=None, id1=None, id2=None,
            period=None, num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r""" For each point in `sample1`, identify all `sample2` points within the input
    `smoothing_scale`; using those points together with the input `weights2`,
    the `inertia_tensor_per_object` function calculates the inertia tensor
    of the mass distribution surrounding each point in `sample1`.

    Parameters
    ----------
    sample1 : array_like
        Numpy array of shape (npts1, 3) storing 3-D positions of points.

        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like
        Numpy array of shape (npts2, 3) storing 3-D positions of the point masses
        used to calculate the inertia tensor of every `sample1` point.

    smoothing_scale : array_like
        Numpy array of shape (npts1, ) storing the three-dimensional distance 
        from each `sample1` point defining which points in `sample2` are used 
        to compute the inertia tensor.  If a single float is passed, that value
        is used for all points in `sample1`.

    weights2 : array_like, optional
        Numpy array of shape (npts2,) storing the mass of each `sample2` point
        used to calculate the inertia tensor of every `sample1` point.
        Default is np.ones(npts2), i.e. equal weight to each point.

    id1 : array_like, optional
        array of integer IDs of shape (npts1, ).  Default is np.ones(npts1).

    id2 : array_like, optional
        array of integer IDs of shape (npts2, ).  Default is np.ones(npts2).

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Default is None, in which case no PBCs will be applied.

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
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for
        ``approx_cell1_size`` for details.

    Returns
    -------
    inertia_tensors : ndarray
        Numpy array of shape (npts1, 3, 3) storing the inertia tensor for every
        object in `sample1`.

    sum_of_masses : ndarray
        Numpy array of shape (npts1, ) storing the sum of the masses of the
        `sample2` points that fall within `smoothing_scale` of each `sample1` point

    Examples
    --------
    >>> npts1, npts2 = 50, 75
    >>> sample1 = np.random.random((npts1, 3))
    >>> sample2 = np.random.random((npts2, 3))
    >>> weights2 = np.random.random(npts2)
    >>> smoothing_scale = 0.1
    >>> result = inertia_tensor_per_object(sample1, sample2, smoothing_scale, weights2)

    Notes
    -----
    For every pair of points, :math:`i, j` in `sample1`, `sample2`,
    the contribution to the inertia tensor is:

    .. math::

        \mathcal{I}_{\rm ij} = m_{\rm j}\begin{bmatrix}
                \delta x_{\rm ij}*\delta x_{\rm ij} & \delta x_{\rm ij}*\delta y_{\rm ij} & \delta x_{\rm ij}*\delta z_{\rm ij} \\
                \delta y_{\rm ij}*\delta x_{\rm ij} & \delta y_{\rm ij}*\delta y_{\rm ij} & \delta y_{\rm ij}*\delta z_{\rm ij} \\
                \delta z_{\rm ij}*\delta x_{\rm ij} & \delta z_{\rm ij}*\delta y_{\rm ij} & \delta z_{\rm ij}*\delta z_{\rm ij}
            \end{bmatrix}

    The :math:`\delta x_{\rm ij}`, :math:`\delta y_{\rm ij}`, and :math:`\delta z_{\rm ij}` terms
    store the coordinate distances between the pair of points
    (optionally accounting for periodic boundary conditions), and :math:`m_{\rm j}` stores
    the mass of the `sample2` point.

    To calculate the inertia tensor :math:`\mathcal{I}_{\rm i}` for the
    :math:`i^{\rm th}` point in `sample1`, the `inertia_tensor_per_object` function
    sums up the contributions :math:`\mathcal{I}_{\rm ij}` for all :math:`j` such that the
    distance between the two points :math:`D_{\rm ij}`
    is less than the smoothing scale :math:`D_{\rm smooth}`:

    .. math::

        \mathcal{I}_{\rm i} = \sum_{j}^{r_{\rm ij} < D_{\rm smooth}} \mathcal{I}_{\rm ij}

    There are several convenience functions available to derive quantities
    from the returned inertia tensors:

        * `~halotools.mock_observables.principal_axes_from_inertia_tensors`
        * `~halotools.mock_observables.sphericity_from_inertia_tensors`
        * `~halotools.mock_observables.triaxility_from_inertia_tensors`
        * `~halotools.mock_observables.axis_ratios_from_inertia_tensors`

    """
    num_threads = get_num_threads(num_threads, enforce_max_cores=False)
    period, PBCs = get_period(period)

    # process arguments
    N1 = np.shape(sample1)[0]
    N2 = np.shape(sample2)[0]

    # extract maximum smoothing scale
    smoothing_scale = np.atleast_1d(smoothing_scale)
    if len(smoothing_scale)==1:
        smoothing_scale = np.array([smoothing_scale]*N1).reshape((N1,))
    msg = "np.shape(smoothing_scale) = {0} should be ({1}, )"
    assert np.shape(smoothing_scale) == (N1,), msg.format(np.shape(smoothing_scale), sample1.shape[0])

    max_smoothing_scale = np.max(smoothing_scale)

    # At this point, period may still be set to None,
    # in which case we must remap our points inside the smallest enclosing cube
    # and set ``period`` equal to this cube size.
    if period is None:
        x1in, y1in, z1in, x2in, y2in, z2in, period = (
            _enclose_in_box(
                sample1[:, 0], sample1[:, 1], sample1[:, 2],
                sample2[:, 0], sample2[:, 1], sample2[:, 2],
                min_size=[max_smoothing_scale*3.0, max_smoothing_scale*3.0, max_smoothing_scale*3.0]))
    else:
        x1in = sample1[:, 0]
        y1in = sample1[:, 1]
        z1in = sample1[:, 2]
        x2in = sample2[:, 0]
        y2in = sample2[:, 1]
        z2in = sample2[:, 2]

    # weights
    if weights2 is None:
        weights2 = np.ones(N2).astype('float')
    else:
        weights2 = np.atleast_1d(weights2).astype('float')

    # particle IDs
    if id1 is None:
        id1 = np.ones(N1).astype('int')
    else:
        id1 = np.atleast_1d(id1).astype('int')
    if id2 is None:
        id2 = np.ones(N2)
    else:
        id2 = np.atleast_1d(id2).astype('int')

    msg = "np.shape(weights2) = {0} should be ({1}, )"
    assert np.shape(weights2) == (sample2.shape[0], ), msg.format(np.shape(weights2), sample2.shape[0])

    msg = "np.shape(id1) = {0} should be ({1}, )"
    assert np.shape(id1) == (sample1.shape[0], ), msg.format(np.shape(id1), sample1.shape[0])
    msg = "np.shape(id2) = {0} should be ({1}, )"
    assert np.shape(id2) == (sample2.shape[0], ), msg.format(np.shape(id2), sample2.shape[0])

    xperiod, yperiod, zperiod = period
    _enforce_maximum_search_length(max_smoothing_scale, period)
    enforce_sample_respects_pbcs(x1in, y1in, z1in, period)
    enforce_sample_respects_pbcs(x2in, y2in, z2in, period)

    search_xlength = max_smoothing_scale
    search_ylength = max_smoothing_scale
    search_zlength = max_smoothing_scale

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

    # Create a function object that has a single argument, for parallelization purposes
    smoothing_scale_sq = smoothing_scale*smoothing_scale
    engine = partial(inertia_tensor_per_object_engine, double_mesh,
        x1in, y1in, z1in, id1, x2in, y2in, z2in, id2, weights2, smoothing_scale_sq)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        tensors = np.array([r[0] for r in result])
        sum_of_masses = np.array([r[1] for r in result])
        tensors = np.sum(tensors, axis=0)
        sum_of_masses = np.sum(sum_of_masses, axis=0)
        pool.close()
    else:
        result = engine(cell1_tuples[0])
        tensors, sum_of_masses = result

    return np.array(tensors), np.array(sum_of_masses)
