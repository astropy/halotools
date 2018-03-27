"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing
from functools import partial

from .engines import reduced_inertia_tensor_per_object_engine

from ..mock_observables_helpers import get_num_threads, get_period, enforce_sample_respects_pbcs
from ..pair_counters.mesh_helpers import (_set_approximate_cell_sizes,
    _cell1_parallelization_indices, _enclose_in_box, _enforce_maximum_search_length)
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh

from ...utils import rotation_matrices_from_vectors

__author__ = ('Andrew Hearin', 'Duncan Campbell')
__all__ = ('inertia_tensor_per_object', )


def reduced_inertia_tensor_per_object(sample1, sample2, smoothing_scale,
            weights2=None, id1=None, id2=None, v1=None, q1=None, s1=None,
            period=None, num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r""" For each point in `sample1`, identify all `sample2` points within the input
    `smoothing_scale` where `id1`  is equal to `id2`; using those points together with
    the input `weights2`, the `inertia_tensor_per_object` function calculates the
    reducded inertia tensor of the mass distribution surrounding each point in `sample1`.

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
        Three-dimensional distance from each `sample1` point defining
        which points in `sample2` are used to compute the inertia tensor

    weights2 : array_like, optional
        Numpy array of shape (npts2,) storing the mass of each `sample2` point
        used to calculate the inertia tensor of every `sample1` point.
        Default is np.ones(npts2).

    id1 : array_like, optional
        array of integer IDs of shape (npts1, ).  Default is np.ones(npts1).

    id2 : array_like, optional
        array of integer IDs of shape (npts2, ).  Default is np.ones(npts2).

    v1 : array_like
        array of principle eigenvectors for `sample1`. The array must be of shape (npts1, 3).
        The default is np.array([1,0,0]*npts1).reshape((npts1,3))

    q1 : array_like, optional
        array of intermediate axis ratios (b/a) of shape (npts1, ). Default is np.ones(npts1).

    s1 : array_like, optional
        array of minor axis ratios (c/a) of shape (npts1, ). Default is np.ones(npts1).

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
    reduced_inertia_tensors : ndarray
        Numpy array of shape (npts1, 3, 3) storing the reduced inertia tensor for every
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
    >>> result = reduced_inertia_tensor_per_object(sample1, sample2, smoothing_scale, weights2)

    Notes
    -----
    The reduced inertia tensor is calculated by pairwise.  For every pair of points,
    :math:`i, j` in `sample1`, `sample2`, the contribution to the reduced inertia tensor is:

    .. math::

        \widetilde{\mathcal{I}}_{\rm ij} = \frac{m_{\rm j}}{r_{\rm ij}^2}\begin{bmatrix}
                \delta x_{\rm ij}\times\delta x_{\rm ij} & \delta x_{\rm ij}\times\delta y_{\rm ij} & \delta x_{\rm ij}\times\delta z_{\rm ij} \\
                \delta y_{\rm ij}\times\delta x_{\rm ij} & \delta y_{\rm ij}\times\delta y_{\rm ij} & \delta y_{\rm ij}\times\delta z_{\rm ij} \\
                \delta z_{\rm ij}\times\delta x_{\rm ij} & \delta z_{\rm ij}\times\delta y_{\rm ij} & \delta z_{\rm ij}\times\delta z_{\rm ij}
            \end{bmatrix}

    The :math:`\delta x_{\rm ij}`, :math:`\delta y_{\rm ij}`, and :math:`\delta z_{\rm ij}` terms
    store the coordinate distances between the pair of points
    (optionally accounting for periodic boundary conditions), :math:`m_{\rm j}` stores
    the mass of the `sample2` point, and :math:`r_{ij}` is the elliptical distance
    in the eigenvector coordinate system bteween the `sample1` and `sample2` point:

    .. math::
        r_{\rm ij} = \sqrt{\delta {x'}_{\rm ij}^2 + \delta {y'}_{\rm ij}^2/q_{\rm i}^2 + \delta {z'}_{\rm ij}^2/s_{\rm i}^2 }

    where, e.g., :math:`\delta {x'}_{\rm ij}` is x-position of the :math:`j^{\rm th}` point in `sample2`
    in the eigenvector coordinate system centered on the :math:`i^{\rm th}` point in `sample1`,
    specified by `v1`.

    To calculate the reduced inertia tensor :math:`\widetilde{\mathcal{I}}_{\rm i}` for the
    :math:`i^{\rm th}` point in `sample1`, the `reduced_inertia_tensor_per_object` function
    sums up the contributions :math:`\widetilde{\mathcal{I}}_{\rm ij}` for all :math:`j` such that the
    distance between the two points :math:`D_{\rm ij}` is less than the smoothing scale :math:`D_{\rm smooth}`,
    and `id1` is equal to `id2`:

    .. math::

        \widetilde{\mathcal{I}}_{\rm i} = \sum_{j}^{D_{\rm ij} < D_{\rm smooth}} \mathcal{I}_{\rm ij}

    There are several convenience functions available to derive quantities
    from the returned inertia tensors:

        * `~halotools.mock_observables.principal_axes_from_inertia_tensors`
        * `~halotools.mock_observables.sphericity_from_inertia_tensors`
        * `~halotools.mock_observables.triaxility_from_inertia_tensors`

    """
    num_threads = get_num_threads(num_threads, enforce_max_cores=False)
    period, PBCs = get_period(period)

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

    # process arguments
    N1 = np.shape(sample1)[0]
    N2 = np.shape(sample2)[0]

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

    # axis ratios
    if q1 is None:
        q1 = np.ones(N1).astype('float')
    else:
        q1 = np.atleast_1d(q1).astype('float')
    if s1 is None:
        s1 = np.ones(N1).astype('float')
    else:
        s1 = np.atleast_1d(s1).astype('float')

    # principle axis
    v0 = np.zeros((N1,3))
    v0[:,0] = 1.0
    if v1 is None:
        v1 = v0
    else:
        v1 = np.atleast_1d(v1)
    assert np.shape(v1) == (N1,3)

    # calculate rotation matrices to tranform sample2 coordinates into
    # the eigenvector coordinate system for each popint in sample1
    rot_m = rotation_matrices_from_vectors(v0,v1)

    msg = "np.shape(weights2) = {0} should be ({1}, )"
    assert np.shape(weights2) == (sample2.shape[0], ), msg.format(np.shape(weights2), sample2.shape[0])

    msg = "np.shape(id1) = {0} should be ({1}, )"
    assert np.shape(id1) == (sample1.shape[0], ), msg.format(np.shape(id1), sample1.shape[0])
    msg = "np.shape(id2) = {0} should be ({1}, )"
    assert np.shape(id2) == (sample2.shape[0], ), msg.format(np.shape(id2), sample2.shape[0])

    msg = "np.shape(q1) = {0} should be ({1}, )"
    assert np.shape(q1) == (sample1.shape[0], ), msg.format(np.shape(q1), sample1.shape[0])
    msg = "np.shape(s1) = {0} should be ({1}, )"
    assert np.shape(s1) == (sample1.shape[0], ), msg.format(np.shape(s1), sample1.shape[0])

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
    engine = partial(reduced_inertia_tensor_per_object_engine, double_mesh,
        x1in, y1in, z1in, id1, q1, s1, x2in, y2in, z2in, id2, weights2, smoothing_scale_sq, rot_m)

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
