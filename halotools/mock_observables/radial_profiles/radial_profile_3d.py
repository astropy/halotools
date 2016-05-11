""" Module containing the `~halotools.mock_observables.radial_profile_3d` function 
used to calculate radial profiles as a function of 3d separation. 
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import multiprocessing
from functools import partial 

from .engines import radial_profile_3d_engine 

from ..pair_counters.npairs_3d import _npairs_3d_process_args
from ..pair_counters.mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. marked_counts/counts

__author__ = ('Andrew Hearin', )
__all__ = ('radial_profile_3d', )

def radial_profile_3d(sample1, sample2, sample2_quantity, rbins, 
    normalize_rbins_by=None, return_counts=False, 
    period=None, verbose=False, num_threads=1,
    approx_cell1_size=None, approx_cell2_size=None):
    """ Function used to calculate the mean value of some quantity in ``sample2`` 
    as a function of 3d distance from the points in ``sample1``. As illustrated 
    in the Examples section below, the ``normalize_rbins_by`` argument allows you to 
    optionally normalize the distances defined by ``rbins`` according to 
    some scaling factor defined by the points in ``sample1``. The documentation below 
    shows how to calculate the mean mass accretion rate of ``sample2`` as a function 
    of the Rvir-normalized halo-centric distance from points in ``sample1``. 

    Parameters 
    -----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the 
        Examples section below, for instructions on how to transform 
        your coordinate position arrays into the 
        format accepted by the ``sample1`` and ``sample2`` arguments.   
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points. 

    sample2_quantity: array_like 
        Length-Npts2 array containing the ``sample2`` quantity whose mean 
        value is being calculated as a function of distance from points in ``sample1``. 

    rbins : array_like
        numpy array of length *Nrbins+1* defining the boundaries of bins in which
        pairs are counted.
        Length units assumed to be in Mpc/h, here and throughout Halotools. 

    normalize_rbins_by : array_like, optional 
        Numpy array of length Npts1 defining how the distance between each pair of points 
        will be normalized. For example, if ``normalize_rbins_by`` is defined to be the 
        virial radius of each point in ``sample1``, then the input ``rbins`` will be 
        re-interpreted as referring to :math:`r / R_{\rm vir}`. Default is None, 
        in which case the input ``rbins`` will be interpreted to be an absolute distance 
        in units of Mpc/h. 

        Pay special attention to units with this argument - the Rockstar default is to 
        return halo catalogs with Rvir in kpc/h units but halo centers in Mpc/h units. 

    return_counts : bool, optional 
        If set to True, `radial_profile_3d` will additionally return the number of 
        pairs in each separation bin. Default is False. 

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions 
        in each dimension. If you instead provide a single scalar, Lbox, 
        period is assumed to be the same in all Cartesian directions. 

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
    --------
    result : array_like 
        Numpy array of length *Nrbins* containing the mean value of 
        ``sample2_quantity`` as a function of 3d distance from the points 
        in ``sample1``. 

    counts : array_like, optional 
        Numpy array of length *Nrbins* containing the number of pairs of 
        points in ``sample1`` and ``sample2`` as a function of 3d distance from the points. 
        Only returned if ``return_counts`` is set to True (default is False). 

    Examples 
    --------
    In this example, we'll select two samples of halos, 
    and calculate how the mass accretion of halos in the second set varies as a function 
    of distance from the halos in the first set. For demonstration purposes we'll use 
    fake halos provided by `~halotools.sim_manager.FakeSim`, but the same syntax works for 
    real halos, and likewise for a mock galaxy catalog. 

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> median_mass = np.median(halocat.halo_table['halo_mvir'])
    >>> sample1_mask = halocat.halo_table['halo_mvir'] > median_mass
    >>> halo_sample1 = halocat.halo_table[sample1_mask]
    >>> halo_sample2 = halocat.halo_table[~sample1_mask]

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack([halo_sample1['halo_x'], halo_sample1['halo_y'], halo_sample1['halo_z']]).T
    >>> sample2 = np.vstack([halo_sample2['halo_x'], halo_sample2['halo_y'], halo_sample2['halo_z']]).T
    >>> dmdt_sample2 = halo_sample2['halo_mass_accretion_rate']

    >>> rbins = np.logspace(-1, 1.5, 15)
    >>> result1 = radial_profile_3d(sample1, sample2, dmdt_sample2, rbins, period=halocat.Lbox)

    The array ``result1`` contains the mean mass accretion rate of halos in ``sample2`` 
    in the bins of distance from halos in ``sample1`` determined by ``rbins``. 

    You can retrieve the number counts in these separation bins as follows:

    >>> result1, counts = radial_profile_3d(sample1, sample2, dmdt_sample2, rbins, period=halocat.Lbox, return_counts=True)

    Now suppose that you wish to calculate the same quantity, but instead as a function of 
    :math:`r / R_{\rm vir}`. In this case, we use the ``normalize_rbins_by`` feature. 
    Defining ``rbins`` as follows will give us 15 separation bins linearly spaced 
    between :math:`\\frac{1}{2}R_{\\rm vir}` and :math:`5R_{\\rm vir}`. 

    >>> rvir = halo_sample1['halo_rvir']
    >>> rbins = np.linspace(0.5, 10, 15) 
    >>> result1 = radial_profile_3d(sample1, sample2, dmdt_sample2, rbins, normalize_rbins_by=rvir, period=halocat.Lbox)

    """

    result = _npairs_3d_process_args(sample1, sample2, rbins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rbins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    rmax = np.max(rbins)
    search_xlength, search_ylength, search_zlength = rmax, rmax, rmax 

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    sample2_quantity, distance_normalization = _radial_profile_3d_process_additional_inputs(
        sample1, sample2, sample2_quantity, rbins, normalize_rbins_by, period)

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(radial_profile_3d_engine, double_mesh, 
        x1in, y1in, z1in, x2in, y2in, z2in, 
        distance_normalization, sample2_quantity, rbins)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)


    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        marked_counts, counts = result
        marked_counts = np.sum(np.array(marked_counts), axis=0)
        counts = np.sum(np.array(counts), axis=0)
        pool.close()
    else:
        marked_counts, counts = engine(cell1_tuples[0])

    marked_counts = np.diff(marked_counts)
    counts = np.diff(counts)

    result = marked_counts/counts

    if return_counts is True:
        return result, counts 
    else:
        return result


def _radial_profile_3d_process_additional_inputs(sample1, sample2, sample2_quantity, 
    normalize_rbins_by, rbins, period):
    """
    """
    msg = "See commented section marked ### LEFT OFF HERE ### "
    raise ValueError(msg)

    npts1 = sample1.shape[0]
    npts2 = sample2.shape[0]
    npts_quantity2 = len(sample2_quantity)
    sample2_quantity = np.atleast_1d(sample2_quantity)
    try:
        assert npts_quantity2 == npts2 
    except AssertionError:
        msg = ("Input ``sample2_quantity`` has %i elements, "
            "but input ``sample2`` has %i elements.\n" % (npts_quantity2, npts2))
        raise ValueError(msg)

    if normalize_rbins_by is None:
        distance_normalization = np.ones(npts1)
    else:
        distance_normalization = np.atleast_1d(normalize_rbins_by)
        npts_normalization = len(distance_normalization)
        try:
            assert npts_normalization == npts1
        except AssertionError:
            msg = ("Input ``normalize_rbins_by`` has %i elements, "
                "but input ``sample1`` has %i elements.\n" % (npts_normalization, npts1))
            raise ValueError(msg)

    ### LEFT OFF HERE ### 
    rmax = np.max(rbins)
    max_rvir = np.max(distance_normalization)
    max_search_radius = max_rvir*rmax
    try:
        minperiod = np.min(np.atleast_1d(period))
        assert max_search_radius < minperiod
    except AssertionError:
        msg = ("You are operating the ``radial_profile_3d`` function "
            "using the ``normalize_rbins_by`` feature.\n"
            "The largest value of the input ``normalize_rbins_by`` array is %.2f.\n"
            "The largest value of the input ``rbins`` array is %.2f.\n"
            "Thus you requested to count pairs over a distance of %.2f,"
            "but your simulation has side length %.2f.\n"
            "It is not permissible to attempt to count pairs across distances \n"
            "greater than Lbox/3 with Halotools. If you need to count pairs across such distances,\n"
            "you should be using a larger simulation.\n" % ())
        raise ValueError(msg)

    return sample2_quantity, distance_normalization
















