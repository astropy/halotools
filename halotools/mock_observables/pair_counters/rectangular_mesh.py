""" Module containing `~halotools.mock_observables.RectangularDoubleMesh`,
the primary data structure used to optimize pairwise
calculations throughout the `~halotools.mock_observables` sub-package.
"""
import numpy as np
from math import floor

__all__ = ('RectangularDoubleMesh', )
__author__ = ('Andrew Hearin', )

default_max_cells_per_dimension_cell1 = 50
default_max_cells_per_dimension_cell2 = 50


def digitized_position(p, cell_size, num_divs):
    """ Function returns a discretized spatial position of input point(s).
    """
    ip = np.floor(p // cell_size).astype(int)
    return np.where(ip >= num_divs, num_divs-1, ip)


def sample1_cell_size(period, search_length, approx_cell_size,
        max_cells_per_dimension=default_max_cells_per_dimension_cell1):
    """ Function determines the size of the cells of mesh1.
    The conditions that must be met are that the cell size must
    be less than the search length, must evenly divide the box length,
    and may not exceed ``max_cells_per_dimension``.
    """
    if search_length > period/3.:
        msg = ("Input ``search_length`` cannot exceed period/3")
        raise ValueError(msg)

    ndivs = int(floor(period/float(approx_cell_size)))
    ndivs = max(ndivs, 1)
    ndivs = min(max_cells_per_dimension, ndivs)

    nsearch = int(floor(period/float(search_length)))
    nsearch = max(nsearch, 1)

    ndivs = min(ndivs, nsearch)
    ndivs = max(3, ndivs)
    cell_size = period/float(ndivs)

    return cell_size


def sample2_cell_sizes(period, sample1_cell_size, approx_cell_size,
        max_cells_per_dimension=default_max_cells_per_dimension_cell2):
    """ Function determines the size of the cells of mesh2.
    The conditions that must be met are that the cell size must
    be less than the search length, must evenly divide the box length,
    and may not exceed ``max_cells_per_dimension``.
    """
    num_sample1_cells = int(np.round(period / sample1_cell_size))
    ndivs_sample1_cells = int(np.round(sample1_cell_size/float(approx_cell_size)))
    ndivs_sample1_cells = max(1, ndivs_sample1_cells)
    ndivs_sample1_cells = min(max_cells_per_dimension, ndivs_sample1_cells)
    num_sample2_cells = num_sample1_cells*ndivs_sample1_cells
    if num_sample2_cells > max_cells_per_dimension:
        num2_per_num1 = max_cells_per_dimension // num_sample1_cells
        num_sample2_cells = num2_per_num1*num_sample1_cells
    cell_size = period/float(num_sample2_cells)
    return cell_size


class RectangularMesh(object):
    """ Underlying mesh structure used to place points into rectangular cells
    within a simulation volume.

    The simulation box is divided into rectanguloid cells whose edges
    and faces are aligned with the Cartesian coordinates of the box.
    Each spatial point in the box belongs to a unique cell.
    Any cell can be identified by either its tuple indices, (ix, iy, iz),
    or by the unique integer ID assigned it via the dictionary ordering
    of tuple indices:

        * (0, 0, 0) <--> 0

        * (0, 0, 1) <--> 1

        * (0, 0, 2) <--> 2

        * ...

        * (0, 1, 0) <--> num_zdivs

        * (0, 1, 1) <--> num_zdivs + 1

        * ...,

    and so forth.

    Each point thus has a unique triplet of integers specifying
    the subvolume containing it, or equivalently a unique integer specifying
    the subvolume containing it, called the *cell_id*.

    """

    def __init__(self, x1in, y1in, z1in, xperiod, yperiod, zperiod,
            approx_xcell_size, approx_ycell_size, approx_zcell_size):
        """
        Parameters
        ----------
        x1in, y1in, z1in : arrays
            Length-*Npts* arrays containing the spatial position of the *Npts* points.

        xperiod, yperiod, zperiod : floats
            Length scale defining the periodic boundary conditions in each dimension.
            In virtually all realistic cases, these are all equal.

        approx_xcell_size, approx_ycell_size, approx_zcell_size : float
            approximate cell sizes into which the simulation box will be divided.
            These are only approximate because in each dimension,
            the actual cell size must be evenly divide the box size.

        Examples
        ---------
        >>> Npts, Lbox = int(1e4), 1000
        >>> xperiod, yperiod, zperiod = Lbox, Lbox, Lbox
        >>> approx_xcell_size = Lbox/10.
        >>> approx_ycell_size = Lbox/10.
        >>> approx_zcell_size = Lbox/10.

        Let's create some fake data to demonstrate the mesh structure:

        >>> from astropy.utils.misc import NumpyRNGContext
        >>> fixed_seed = 43
        >>> with NumpyRNGContext(fixed_seed): pos = np.random.uniform(0, Lbox, 3*Npts).reshape(Npts, 3)
        >>> x, y, z = pos[:,0], pos[:,1], pos[:,2]
        >>> mesh = RectangularMesh(x, y, z, xperiod, yperiod, zperiod, approx_xcell_size, approx_ycell_size, approx_zcell_size)

        Since we used approximate cell sizes *Lbox/10* that
        exactly divided the period in each dimension,
        then we know there are *10* subvolumes-per-dimension.
        So, for example, based on the discussion above,
        *cellID = 0* will correspond to *cell_tupleID = (0, 0, 0)*,
        *cellID = 5* will correspond to *cell_tupleID = (0, 0, 5)* and
        *cellID = 13* will correspond to *cell_tupleID = (0, 1, 3).*

        Now that your mesh has been built, you can efficiently access
        the *x, y, z* positions of the points lying in
        the subvolume with *cellID = i* as follows:

        >>> i = 13
        >>> ith_subvol_first, ith_subvol_last = mesh.cell_id_indices[i], mesh.cell_id_indices[i+1]
        >>> xcoords_ith_subvol = x[mesh.idx_sorted][ith_subvol_first:ith_subvol_last]
        >>> ycoords_ith_subvol = y[mesh.idx_sorted][ith_subvol_first:ith_subvol_last]
        >>> zcoords_ith_subvol = z[mesh.idx_sorted][ith_subvol_first:ith_subvol_last]

        """

        self.npts = x1in.shape[0]

        self.xperiod = xperiod
        self.yperiod = yperiod
        self.zperiod = zperiod

        self.num_xdivs = max(int(np.round(xperiod / approx_xcell_size)), 1)
        self.num_ydivs = max(int(np.round(yperiod / approx_ycell_size)), 1)
        self.num_zdivs = max(int(np.round(zperiod / approx_zcell_size)), 1)
        self.ncells = self.num_xdivs*self.num_ydivs*self.num_zdivs

        self.xcell_size = self.xperiod / float(self.num_xdivs)
        self.ycell_size = self.yperiod / float(self.num_ydivs)
        self.zcell_size = self.zperiod / float(self.num_zdivs)

        ix = digitized_position(x1in, self.xcell_size, self.num_xdivs)
        iy = digitized_position(y1in, self.ycell_size, self.num_ydivs)
        iz = digitized_position(z1in, self.zcell_size, self.num_zdivs)

        cell_ids = self.cell_id_from_cell_tuple(ix, iy, iz)
        self.idx_sorted = np.ascontiguousarray(np.argsort(cell_ids))

        cell_id_indices = np.searchsorted(cell_ids, np.arange(self.ncells),
            sorter=self.idx_sorted)
        cell_id_indices = np.append(cell_id_indices, self.npts)
        self.cell_id_indices = np.ascontiguousarray(cell_id_indices)

    def cell_id_from_cell_tuple(self, ix, iy, iz):
        return ix*(self.num_ydivs*self.num_zdivs) + iy*self.num_zdivs + iz


class RectangularDoubleMesh(object):
    """ Fundamental data structure of the `~halotools.mock_observables` sub-package.
    `~halotools.mock_observables.RectangularDoubleMesh` is built up from two instances
    of `~halotools.mock_observables.pair_counters.rectangular_mesh.RectangularMesh`.
    """

    def __init__(self, x1, y1, z1, x2, y2, z2,
            approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
            approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
            search_xlength, search_ylength, search_zlength,
            xperiod, yperiod, zperiod, PBCs=True,
            max_cells_per_dimension_cell1=default_max_cells_per_dimension_cell1,
            max_cells_per_dimension_cell2=default_max_cells_per_dimension_cell2):
        """
        Parameters
        ----------
        x1, y1, z1 : arrays
            Length-*Npts1* arrays containing the spatial position of the *Npts1* points.

        x2, y2, z2 : arrays
            Length-*Npts2* arrays containing the spatial position of the *Npts2* points.

        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size : float
            approximate cell sizes into which the simulation box will be divided.
            These are only approximate because in each dimension,
            the actual cell size must be evenly divide the box size.

        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size : float
            An entirely separate tree is built for the *Npts2* points, the structure of
            which is dependent on the struture of the *Npts1* tree as described below.

        search_xlength, search_ylength, search_zlength, floats, optional
            Maximum length over which a pair of points will searched for.
            For example, if using `~halotools.mock_observables.pair_counters.RectangularDoubleMesh`
            to compute a 3-D correlation function with radial separation bins
            *rbins = [0.1, 1, 10, 25]*, then in this case
            all the search lengths will equal 25.
            If using `~halotools.mock_observables.pair_counters.RectangularDoubleMesh`
            in a projected correlation function with *rp_bins = [0.1, 1, 10, 25]* and
            *pi_max = 40*, then *search_xlength = search_ylength = 25* and
            *search_zlength = 40*.

        xperiod, yperiod, zperiod : floats
            Length scale defining the periodic boundary conditions in each dimension.
            In virtually all realistic cases, these are all equal.

        PBCs : bool, optional
            Boolean specifying whether or not the box has periodic boundary conditions.
            Default is True.

        max_cells_per_dimension_cell1 : int, optional
            Maximum number of cells per dimension. Default is 50.

        max_cells_per_dimension_cell2 : int, optional
            Maximum number of cells per dimension. Default is 50.

        """
        self.xperiod = xperiod
        self.yperiod = yperiod
        self.zperiod = zperiod
        self.search_xlength = search_xlength
        self.search_ylength = search_ylength
        self.search_zlength = search_zlength
        self._PBCs = PBCs

        self._check_sensible_constructor_inputs()

        approx_x1cell_size = sample1_cell_size(xperiod, search_xlength, approx_x1cell_size,
            max_cells_per_dimension=max_cells_per_dimension_cell1)
        approx_y1cell_size = sample1_cell_size(yperiod, search_ylength, approx_y1cell_size,
            max_cells_per_dimension=max_cells_per_dimension_cell1)
        approx_z1cell_size = sample1_cell_size(zperiod, search_zlength, approx_z1cell_size,
                max_cells_per_dimension=max_cells_per_dimension_cell1)
        self.mesh1 = RectangularMesh(x1, y1, z1, xperiod, yperiod, zperiod,
            approx_x1cell_size, approx_y1cell_size, approx_z1cell_size)

        approx_x2cell_size = sample2_cell_sizes(xperiod, self.mesh1.xcell_size, approx_x2cell_size,
            max_cells_per_dimension=max_cells_per_dimension_cell2)
        approx_y2cell_size = sample2_cell_sizes(yperiod, self.mesh1.ycell_size, approx_y2cell_size,
            max_cells_per_dimension=max_cells_per_dimension_cell2)
        approx_z2cell_size = sample2_cell_sizes(zperiod, self.mesh1.zcell_size, approx_z2cell_size,
            max_cells_per_dimension=max_cells_per_dimension_cell2)
        self.mesh2 = RectangularMesh(x2, y2, z2, xperiod, yperiod, zperiod,
            approx_x2cell_size, approx_y2cell_size, approx_z2cell_size)

        self.num_xcell2_per_xcell1 = self.mesh2.num_xdivs // self.mesh1.num_xdivs
        self.num_ycell2_per_ycell1 = self.mesh2.num_ydivs // self.mesh1.num_ydivs
        self.num_zcell2_per_zcell1 = self.mesh2.num_zdivs // self.mesh1.num_zdivs

    def _check_sensible_constructor_inputs(self):
        try:
            assert self.search_xlength <= self.xperiod/3.
        except AssertionError:
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "You tried to search for pairs out to a length of search_xlength = %.2f,\n"
                "but the size of your box in this dimension is xperiod = %.2f.\n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n" % (self.search_xlength, self.xperiod))
            raise ValueError(msg)

        try:
            assert self.search_ylength <= self.yperiod/3.
        except AssertionError:
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "You tried to search for pairs out to a length of search_ylength = %.2f,\n"
                "but the size of your box in this dimension is yperiod = %.2f.\n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n" % (self.search_ylength, self.yperiod))
            raise ValueError(msg)

        try:
            assert self.search_zlength <= self.zperiod/3.
        except AssertionError:
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "You tried to search for pairs out to a length of search_zlength = %.2f,\n"
                "but the size of your box in this dimension is zperiod = %.2f.\n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n" % (self.search_zlength, self.zperiod))
            raise ValueError(msg)
