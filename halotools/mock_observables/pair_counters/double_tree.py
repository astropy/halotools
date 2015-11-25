# -*- coding: utf-8 -*-

"""
Data structures used for efficient, cache-aware pairwise calculations on simulation boxes.
"""

import numpy as np
from math import ceil, floor
from ...custom_exceptions import *

__all__=['FlatRectanguloidTree', 'FlatRectanguloidDoubleTree']
__author__ = ['Andrew Hearin', 'Duncan Campbell']

class FlatRectanguloidTree(object):
    """ Flat, rectangular tree structure for the simulation box used by the `~halotools.mock_observables` sub-package. 

    The simulation box is divided into rectanguloid cells whose edges 
    and faces are aligned with the x-, y- and z-axes of the box. 
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

    Each point thus has a unique triplet specifying 
    the subvolume containing it, referred to as the 
    *cell_tupleIDs* of that point. And equivalently, 
    And equivalently, each point has a unique integer specifying 
    the subvolume containing it, called the *cellID*. 

    """

    def __init__(self, x, y, z, 
        approx_xcell_size, approx_ycell_size, approx_zcell_size, 
        xperiod, yperiod, zperiod):
        """
        Parameters 
        ----------
        x, y, z : arrays
            Length-*Npts* arrays containing the spatial position of the *Npts* points. 

        approx_xcell_size, approx_ycell_size, approx_zcell_size : float 
            approximate cell sizes into which the simulation box will be divided. 
            These are only approximate because in each dimension, 
            the actual cell size must be evenly divide the box size. 

        xperiod, yperiod, zperiod : floats
            Length scale defining the periodic boundary conditions in each dimension. 
            In virtually all realistic cases, these are all equal. 

        Examples 
        ---------
        >>> Npts, Lbox = 1e4, 1000
        >>> xperiod, yperiod, zperiod = Lbox, Lbox, Lbox
        >>> approx_xcell_size = Lbox/10.
        >>> approx_ycell_size = Lbox/10.
        >>> approx_zcell_size = Lbox/10.

        Let's create some fake data to demonstrate the tree structure:

        >>> np.random.seed(43)
        >>> x = np.random.uniform(0, Lbox, Npts)
        >>> y = np.random.uniform(0, Lbox, Npts) 
        >>> z = np.random.uniform(0, Lbox, Npts) 
        >>> tree = FlatRectanguloidTree(x, y, z, approx_xcell_size, approx_ycell_size, approx_zcell_size, xperiod, yperiod, zperiod)

        Since we used approximate cell sizes *Lbox/10* that 
        exactly divided the period in each dimension, 
        then we know there are *10* subvolumes-per-dimension. 
        So, for example, based on the discussion above, 
        *cellID = 0* will correspond to *cell_tupleID = (0, 0, 0)*,  
        *cellID = 5* will correspond to *cell_tupleID = (0, 0, 5)* and 
        *cellID = 13* will correspond to *cell_tupleID = (0, 1, 3).* 

        Now that your tree has been built, you can efficiently access 
        the *x, y, z* positions of the points lying in 
        the subvolume with *cellID = i* as follows:

        >>> i = 13
        >>> ith_subvol_slice = tree.slice_array[i]
        >>> xcoords_ith_subvol = tree.x[ith_subvol_slice]
        >>> ycoords_ith_subvol = tree.y[ith_subvol_slice]
        >>> zcoords_ith_subvol = tree.z[ith_subvol_slice]

        """

        self._check_sensible_constructor_inputs()

        self.xperiod = xperiod 
        self.yperiod = yperiod 
        self.zperiod = zperiod 

        self.num_xdivs = int(round(xperiod/float(approx_xcell_size)))
        self.num_ydivs = int(round(yperiod/float(approx_ycell_size)))
        self.num_zdivs = int(round(zperiod/float(approx_zcell_size)))

        self.xcell_size = self.xperiod/float(self.num_xdivs)
        self.ycell_size = self.yperiod/float(self.num_ydivs)
        self.zcell_size = self.zperiod/float(self.num_zdivs)
        
        # Build the tree
        idx_sorted, slice_array = self.compute_cell_structure(x, y, z)
        self.x = np.ascontiguousarray(x[idx_sorted], dtype=np.float64)
        self.y = np.ascontiguousarray(y[idx_sorted], dtype=np.float64)
        self.z = np.ascontiguousarray(z[idx_sorted], dtype=np.float64)
        self.slice_array = slice_array
        self.idx_sorted = idx_sorted

    def _check_sensible_constructor_inputs(self):
        """
        """
        pass

    def cell_idx_from_cell_tuple(self, ix, iy, iz):
        """ Return the *cellID* from the *cell_tupleIDs*. 

        Parameters 
        -----------
        ix, iy, iz : int 
            Integers providing the *cell_tupleIDs* of the points. 

        Returns 
        ---------
        cellID : int 
            Integers providing the corresponding *cellIDs*. 
        """
        return ix*(self.num_ydivs*self.num_zdivs) + iy*self.num_zdivs + iz

    def cell_tuple_from_cell_idx(self, cellID):
        """ Return the *cell_tupleIDs* from the *cellID*. 

        Parameters 
        -----------
        cellID : int 
            Integers providing the corresponding *cellIDs*. 

        Returns 
        ---------
        ix, iy, iz : int 
            Integers providing the *cell_tupleIDs* of the points. 
        """
        
        nxny = self.num_ydivs*self.num_zdivs
        
        ix = cellID / nxny
        
        iy = (cellID - ix*nxny) / self.num_zdivs
        
        iz = cellID - (ix*self.num_ydivs*self.num_zdivs) - (iy*self.num_zdivs)
        
        return ix, iy, iz

    def compute_cell_structure(self, x, y, z):
        """ 
        Method divides the periodic box into rectangular subvolumes, and assigns a 
        subvolume index to each point.  The returned arrays can be used to efficiently 
        access only those points in a given subvolume. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-*Npts* arrays containing the spatial position of the *Npts* points. 

        Returns 
        -------
        idx_sorted : array_like 
            Array of indices that sort the points according to the dictionary 
            order of the 3d subvolumes. 

        slice_array : array_like 
            array of `slice` objects used to access the elements of the *x, y, z* 
            values of points residing in a given subvolume. 
        """

        ix = np.floor(x/self.xcell_size).astype(int)
        iy = np.floor(y/self.ycell_size).astype(int)
        iz = np.floor(z/self.zcell_size).astype(int)
        
        #take care of points right on the boundary
        ix = np.where(ix >= self.num_xdivs, self.num_xdivs-1, ix)
        iy = np.where(iy >= self.num_ydivs, self.num_ydivs-1, iy)
        iz = np.where(iz >= self.num_zdivs, self.num_zdivs-1, iz)

        cell_idx_of_particles = self.cell_idx_from_cell_tuple(ix, iy, iz)
        
        num_total_cells = self.num_xdivs*self.num_ydivs*self.num_zdivs

        idx_sorted = np.argsort(cell_idx_of_particles)
        bin_indices = np.searchsorted(cell_idx_of_particles[idx_sorted], 
            np.arange(num_total_cells))
        bin_indices = np.append(bin_indices, None)
        
        slice_array = np.empty(num_total_cells, dtype=object)
        for icell in xrange(num_total_cells):
            slice_array[icell] = slice(bin_indices[icell], bin_indices[icell+1], 1)
            
        return idx_sorted, slice_array


class FlatRectanguloidDoubleTree(object): 
    """ Double tree structure built up from two instances of `~halotools.mock_observables.pair_counters.FlatRectanguloidTree` used by the `~halotools.mock_observables` sub-package. 
    """

    def __init__(self, x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        search_xlength, search_ylength, search_zlength, 
        xperiod, yperiod, zperiod, PBCs = True):
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
            For example, if using `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
            to compute a 3-D correlation function with radial separation bins 
            *rbins = [0.1, 1, 10, 25]*, then in this case 
            all the search lengths will equal 25. 
            If using `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
            in a projected correlation function with *rp_bins = [0.1, 1, 10, 25]* and 
            *pi_max = 40*, then *search_xlength = search_ylength = 25* and 
            *search_zlength = 40*. 

        xperiod, yperiod, zperiod : floats
            Length scale defining the periodic boundary conditions in each dimension. 
            In virtually all realistic cases, these are all equal. 

        PBCs : bool, optional 
            Boolean specifying whether or not the box has periodic boundary conditions. 
            Default is True. 
        """


        self.xperiod = xperiod 
        self.yperiod = yperiod 
        self.zperiod = zperiod 
        self.search_xlength = search_xlength 
        self.search_ylength = search_ylength 
        self.search_zlength = search_zlength 
        self._PBCs = PBCs

        self._check_sensible_constructor_inputs()

        # Set the cell size of sample 1
        modified_x1_cellsize, modified_y1_cellsize, modified_z1_cellsize = (
            self.set_tree1_cell_sizes(
                approx_x1cell_size, approx_y1cell_size, approx_z1cell_size)
            )

        # Build the tree for sample 1
        self.tree1 = FlatRectanguloidTree(x1, y1, z1, 
            modified_x1_cellsize, modified_y1_cellsize, modified_z1_cellsize, 
            xperiod, yperiod, zperiod)

        # Define a few convenient pointers
        self.num_x1divs = self.tree1.num_xdivs 
        self.num_y1divs = self.tree1.num_ydivs 
        self.num_z1divs = self.tree1.num_zdivs 

        self.x1cell_size = self.tree1.xcell_size
        self.y1cell_size = self.tree1.ycell_size
        self.z1cell_size = self.tree1.zcell_size

        # Set the cell size of sample 2
        modified_x2_cellsize, modified_y2_cellsize, modified_z2_cellsize = (
            self.set_tree2_cell_sizes(
                approx_x2cell_size, approx_y2cell_size, approx_z2cell_size)
            )

        # Build the tree for sample 2
        self.tree2 = FlatRectanguloidTree(x2, y2, z2, 
            modified_x2_cellsize, modified_y2_cellsize, modified_z2_cellsize, 
            xperiod, yperiod, zperiod)

        # Define a few convenient pointers
        self.num_x2divs = self.tree2.num_xdivs 
        self.num_y2divs = self.tree2.num_ydivs 
        self.num_z2divs = self.tree2.num_zdivs 

        self.x2cell_size = self.tree2.xcell_size
        self.y2cell_size = self.tree2.ycell_size
        self.z2cell_size = self.tree2.zcell_size

        self.num_xcell2_per_xcell1 = self.num_x2divs/self.num_x1divs
        self.num_ycell2_per_ycell1 = self.num_y2divs/self.num_y1divs
        self.num_zcell2_per_zcell1 = self.num_z2divs/self.num_z1divs

    def _check_sensible_constructor_inputs(self):
        """
        """
        try:
            assert self.search_xlength <= self.xperiod/3.
            assert self.search_ylength <= self.yperiod/3.
            assert self.search_zlength <= self.zperiod/3.
        except AssertionError:
            msg = ("\n The maximum length over which you search for pairs of points \n"
                "cannot be larger than Lbox/3 in any dimension. \n"
                "If you need to count pairs on these length scales, \n"
                "you should use a larger simulation.\n")
            raise HalotoolsError(msg)
        


    def set_tree1_cell_sizes(self, 
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        max_cells_per_dimension = 25):
        """ Method sets the size of the cells of the first tree structure. 

        In each dimension, the cell size is required to evenly divide the 
        size of the box enclosing the points. In order for the bookkeeping 
        of the `adjacent_cell_generator` method to be correct, 
        there must be at least three cell-lengths per box-length. 
        Furthermore, there should not be too many cells per dimension 
        or performance suffers, so the ``max_cells_per_dimension`` argument 
        limits how small the cell-lengths can be. 
        
        Parameters 
        -----------
        approx_x1cell_size : float 
            Rough estimate for cell-length in the x-dimension. The exact 
            cell-length will be adjusted to meet the criteria described above. 

        approx_y1cell_size : float 

        approx_z1cell_size : float 

        max_cells_per_dimension : int, optional 

        Returns 
        -------
        modified_x1_cellsize : float 
            Exact size of the cell-length in the x-dimension. 

        modified_y1_cellsize : float 

        modified_z1_cellsize : float 
        """

        x1mult = int(floor(self.xperiod/float(approx_x1cell_size)))
        y1mult = int(floor(self.yperiod/float(approx_y1cell_size)))
        z1mult = int(floor(self.zperiod/float(approx_z1cell_size)))

        x1mult = min(max_cells_per_dimension, x1mult)
        y1mult = min(max_cells_per_dimension, y1mult)
        z1mult = min(max_cells_per_dimension, z1mult)

        xsearch_mult = int(floor(self.xperiod/float(self.search_xlength)))
        ysearch_mult = int(floor(self.yperiod/float(self.search_ylength)))
        zsearch_mult = int(floor(self.zperiod/float(self.search_zlength)))

        x1mult = min(x1mult, xsearch_mult)
        y1mult = min(y1mult, ysearch_mult)
        z1mult = min(z1mult, zsearch_mult)

        x1mult = max(3, x1mult)
        y1mult = max(3, y1mult)
        z1mult = max(3, z1mult)

        modified_x1_cellsize = self.xperiod/float(x1mult)
        modified_y1_cellsize = self.yperiod/float(y1mult)
        modified_z1_cellsize = self.zperiod/float(z1mult)

        return modified_x1_cellsize, modified_y1_cellsize, modified_z1_cellsize

    def set_tree2_cell_sizes(self, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        max_cells_per_dimension = 25):
        """ Method sets the size of the cells of the second tree structure. 

        In each dimension, the cell size is required to evenly divide the 
        cell-size of the corresponding dimension of the first tree structure. 
        In order for the bookkeeping of the `adjacent_cell_generator` method 
        to be correct, in each dimension the cell-lengths of tree 2 cannot 
        be larger than the cell-length of tree 1. 
        Furthermore, there should not be too many cells per dimension 
        or performance suffers, so the ``max_cells_per_dimension`` argument 
        limits how small the cell-lengths can be. 

        Parameters 
        -----------
        approx_x2cell_size : float 
            Rough estimate for cell-length in the x-dimension. The exact 
            cell-length will be adjusted to meet the criteria described above. 

        approx_y2cell_size : float 

        approx_z2cell_size : float 

        max_cells_per_dimension : int, optional 

        Returns 
        -------
        modified_x2_cellsize : float 
            Exact size of the cell-length in the x-dimension. 

        modified_y2_cellsize : float 

        modified_z2_cellsize : float 
        """

        x2mult = int(round(self.tree1.xcell_size/float(approx_x2cell_size)))
        y2mult = int(round(self.tree1.ycell_size/float(approx_y2cell_size)))
        z2mult = int(round(self.tree1.zcell_size/float(approx_z2cell_size)))

        x2mult = max(1, x2mult)
        y2mult = max(1, y2mult)
        z2mult = max(1, z2mult)

        x2mult = min(max_cells_per_dimension, x2mult)
        y2mult = min(max_cells_per_dimension, y2mult)
        z2mult = min(max_cells_per_dimension, z2mult)

        modified_x2_cellsize = self.tree1.xcell_size/x2mult
        modified_y2_cellsize = self.tree1.ycell_size/y2mult
        modified_z2_cellsize = self.tree1.zcell_size/z2mult

        return modified_x2_cellsize, modified_y2_cellsize, modified_z2_cellsize


    def leftmost_cell_tuple_idx2(self, tuple_idx1, num_cell2_per_cell1):
        """ Given a *cell_tupleID1* for tree 1 in a single dimension, 
        and given the number of tree-2 cells per tree1 cells in that dimension, 
        return the *cell_tupleID2* of the tree-2 cell that is flush against the 
        leftmost edge of the tree-1 cell with *cell_tupleID1*. 

        Parameters 
        ------------
        tuple_idx1 : int 
            Index of the *cell_tupleID1* for tree 1 in a single dimension

        num_cell2_per_cell1 : int 
            Number of tree-2 cells per tree-1 cell in the given dimension 

        Returns 
        --------
        result : int 

        """
        return tuple_idx1*num_cell2_per_cell1

    def rightmost_cell_tuple_idx2(self, tuple_idx1, num_cell2_per_cell1):
        """ Given a *cell_tupleID1* for tree 1 in a single dimension, 
        and given the number of tree-2 cells per tree1 cells in that dimension, 
        return the *cell_tupleID2* of the tree-2 cell that is flush against the 
        rightmost edge of the tree-1 cell with *cell_tupleID1*. 

        Parameters 
        ------------
        tuple_idx1 : int 
            Index of the *cell_tupleID1* for tree 1 in a single dimension

        num_cell2_per_cell1 : int 
            Number of tree-2 cells per tree-1 cell in the given dimension 

        Returns 
        --------
        result : int 
        """
        return (tuple_idx1+1)*num_cell2_per_cell1 - 1

    def num_cells_to_cover_search_length(self, cell_size, search_length):
        """ Number of tree-2 cells that need to be searched to the left and right 
        of the current tree-1 cell in order to guarantee that all points 
        inside the search length will be found.
        """
        return int(np.ceil(search_length/float(cell_size)))

    def nonPBC_generator(self, dim_idx, num_covering_steps, num_cell2_per_cell1):
        """
        """
        leftmost_tuple_idx = (
            self.leftmost_cell_tuple_idx2(dim_idx, num_cell2_per_cell1) - 
            num_covering_steps)

        rightmost_tuple_idx = (
            self.rightmost_cell_tuple_idx2(dim_idx, num_cell2_per_cell1) + 
            num_covering_steps)

        return xrange(leftmost_tuple_idx, rightmost_tuple_idx+1)


    def adjacent_cell_generator(self, icell1, 
        search_xlength, search_ylength, search_zlength):
        """
        """


        # Determine the cell tuple from the input cellID
        ix1, iy1, iz1 = self.tree1.cell_tuple_from_cell_idx(icell1)

        # Determine the number of d2 cells are necessary to guarantee that 
        ### we search for all points within the search length 
        num_x2_covering_steps = self.num_cells_to_cover_search_length(
            self.x2cell_size, search_xlength) 

        num_y2_covering_steps = self.num_cells_to_cover_search_length(
            self.y2cell_size, search_ylength)    

        num_z2_covering_steps = self.num_cells_to_cover_search_length(
            self.z2cell_size, search_zlength)  


        # For each spatial dimension, retrieve a generator that yields 
        ### the tuple indices of the adjacent cells that 
        ### we need to search for pairs 
        nonPBC_ix2_generator = self.nonPBC_generator(ix1, 
            num_x2_covering_steps, self.num_xcell2_per_xcell1)

        nonPBC_iy2_generator = self.nonPBC_generator(iy1, 
            num_y2_covering_steps, self.num_ycell2_per_ycell1)

        nonPBC_iz2_generator = self.nonPBC_generator(iz1, 
            num_z2_covering_steps, self.num_zcell2_per_zcell1)

        for nonPBC_ix2 in nonPBC_ix2_generator:
            # determine the pre-shift in the x dimension
            if nonPBC_ix2 < 0:
                x2shift = -self.xperiod
            elif nonPBC_ix2 >= self.num_x2divs:
                x2shift = +self.xperiod
            else:
                x2shift = 0.
            # Now apply the PBCs
            ix2 = nonPBC_ix2 % self.num_x2divs

            for nonPBC_iy2 in nonPBC_iy2_generator:
                # determine the pre-shift in the y dimension
                if nonPBC_iy2 < 0:
                    y2shift = -self.yperiod
                elif nonPBC_iy2 >= self.num_y2divs:
                    y2shift = +self.yperiod
                else:
                    y2shift = 0.
                # Now apply the PBCs
                iy2 = nonPBC_iy2 % self.num_y2divs

                for nonPBC_iz2 in nonPBC_iz2_generator:
                    # determine the pre-shift in the z dimension
                    if nonPBC_iz2 < 0:
                        z2shift = -self.zperiod
                    elif nonPBC_iz2 >= self.num_z2divs:
                        z2shift = +self.zperiod
                    else:
                        z2shift = 0.
                    # Now apply the PBCs
                    iz2 = nonPBC_iz2 % self.num_z2divs

                    # Get icell2 from ix2, iy2, iz2
                    icell2 = self.tree2.cell_idx_from_cell_tuple(ix2, iy2, iz2)

                    yield icell2, x2shift*self._PBCs, y2shift*self._PBCs, z2shift*self._PBCs
                    
                    