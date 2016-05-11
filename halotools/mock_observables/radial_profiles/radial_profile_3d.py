""" Module containing the `~halotools.mock_observables.radial_profile_3d` function 
used to calculate radial profiles as a function of 3d separation. 
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import multiprocessing
from functools import partial 

from .radial_profiles_helpers import (bounds_check_sample2_quantity, 
    get_normalized_rbins, enforce_maximum_search_length_3d)
from .engines import radial_profile_3d_engine 

from ..mock_observables_helpers import get_num_threads, get_period, enforce_sample_respects_pbcs
from ..pair_counters.mesh_helpers import (_set_approximate_cell_sizes, 
    _cell1_parallelization_indices, _enclose_in_box)
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. marked_counts/counts

__author__ = ('Andrew Hearin', )
__all__ = ('radial_profile_3d', )

def radial_profile_3d(sample1, sample2, sample2_quantity, 
    rbins_absolute=None, rbins_normalized=None, normalize_rbins_by=None, 
    return_counts=False, period=None, num_threads=1,
    approx_cell1_size=None, approx_cell2_size=None):
    """ Function used to calculate the mean value of some quantity in ``sample2`` 
    as a function of 3d distance from the points in ``sample1``. 

    As illustrated in the Examples section below, 
    the ``normalize_rbins_by`` argument allows you to 
    optionally normalize the distances defined by ``rbins`` according to 
    some scaling factor defined by the points in ``sample1``. The documentation below 
    shows how to calculate the mean mass accretion rate of ``sample2`` as a function 
    of the Rvir-normalized halo-centric distance from points in ``sample1``. 

    Note that this function can also be used to calculate simply the radial profiles 
    of number counts of ``sample2`` objects as a function of halo-centric distance 
    from ``sample1`` points. If you are only interested in number counts, 
    you can pass in any dummy array for the input ``sample2_quantity``, 
    and set the ``return_counts`` argument to True. 

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
        re-interpreted as referring to :math:`r / R_{\\rm vir}`. Default is None, 
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

    >>> rbins_absolute = np.logspace(-1, 1.5, 15)
    >>> result1 = radial_profile_3d(sample1, sample2, dmdt_sample2, rbins_absolute=rbins_absolute, period=halocat.Lbox)

    The array ``result1`` contains the mean mass accretion rate of halos in ``sample2`` 
    in the bins of distance from halos in ``sample1`` determined by ``rbins_absolute``. 

    You can retrieve the number counts in these separation bins as follows:

    >>> result1, counts = radial_profile_3d(sample1, sample2, dmdt_sample2, rbins_absolute=rbins_absolute, period=halocat.Lbox, return_counts=True)

    Now suppose that you wish to calculate the same quantity, 
    but instead as a function of :math:`r / R_{\rm vir}`. 
    In this case, we use the ``rbins_normalized`` and ``normalize_rbins_by`` arguments. 
    The following choices for these arguments will give us 15 separation bins linearly spaced 
    between :math:`\\frac{1}{2}R_{\\rm vir}` and :math:`5R_{\\rm vir}`. 

    >>> rvir = halo_sample1['halo_rvir']
    >>> rbins_normalized = np.linspace(0.5, 10, 15) 
    >>> result1 = radial_profile_3d(sample1, sample2, dmdt_sample2, rbins_normalized=rbins_normalized, normalize_rbins_by=rvir, period=halocat.Lbox)

    """

    num_threads = get_num_threads(num_threads, enforce_max_cores = False)

    rbins_normalized, normalize_rbins_by = get_normalized_rbins(
        rbins_absolute, rbins_normalized, normalize_rbins_by, sample1)
    squared_normalize_rbins_by = normalize_rbins_by*normalize_rbins_by
    max_rbins_absolute = np.amax(rbins_normalized)*np.amax(normalize_rbins_by)

    period, PBCs = get_period(period)
    # At this point, period may still be set to None, 
    # in which case we must remap our points inside the smallest enclosing cube 
    # and set ``period`` equal to this cube size.
    if period is None:
        x1in, y1in, z1in, x2in, y2in, z2in, period = (
            _enclose_in_box(
                sample1[:,0], sample1[:,2], sample1[:,2], 
                sample2[:,0], sample2[:,2], sample2[:,2], 
                min_size=[max_rbins_absolute*3.0,max_rbins_absolute*3.0,max_rbins_absolute*3.0]))
    else:
        x1in = sample1[:,0]
        y1in = sample1[:,1]
        z1in = sample1[:,2]
        x2in = sample2[:,0]
        y2in = sample2[:,1]
        z2in = sample2[:,2]
    xperiod, yperiod, zperiod = period 
    enforce_maximum_search_length_3d(rbins_normalized, normalize_rbins_by, period)

    enforce_sample_respects_pbcs(x1in, y1in, z1in, period)
    enforce_sample_respects_pbcs(x2in, y2in, z2in, period)

    search_xlength = max_rbins_absolute
    search_ylength = max_rbins_absolute
    search_zlength = max_rbins_absolute

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    sample2_quantity = bounds_check_sample2_quantity(sample2, sample2_quantity)

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(radial_profile_3d_engine, double_mesh, 
        x1in, y1in, z1in, x2in, y2in, z2in, 
        squared_normalize_rbins_by, sample2_quantity, rbins_normalized)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    # print(rbins_normalized)
    # print(set(normalize_rbins_by))

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














