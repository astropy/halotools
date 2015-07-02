# -*- coding: utf-8 -*-

"""
cell structure object used for efficient pairwise operations on simulation boxes.
"""

from __future__ import print_function, division
import numpy as np

__all__=['rect_cuboid_cells']
__author__ = ['Andrew Hearin, Duncan Campbell']

class rect_cuboid_cells():

    def __init__(self, x, y, z, Lbox, cell_size):
        """
        Initialize the grid. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 
        
        Lbox : float
            Length scale defining the periodic boundary conditions

        cell_size : float 
            The approximate cell size into which the box will be divided. 
        """

        self.cell_size = cell_size.astype(np.float)
        self.Lbox = Lbox.astype(np.float)
        self.num_divs = np.floor(Lbox/cell_size).astype(int)
        self.dL = Lbox/self.num_divs
        
        #build grid tree
        idx_sorted, slice_array = self.compute_cell_structure(x, y, z)
        self.x = np.ascontiguousarray(x[idx_sorted],dtype=np.float64)
        self.y = np.ascontiguousarray(y[idx_sorted],dtype=np.float64)
        self.z = np.ascontiguousarray(z[idx_sorted],dtype=np.float64)
        self.slice_array = slice_array
        self.idx_sorted = idx_sorted

    def compute_cell_structure(self, x, y, z):
        """ 
        Method divides the periodic box into regular, cubical subvolumes, and assigns a 
        subvolume index to each point.  The returned arrays can be used to efficiently 
        access only those points in a given subvolume. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 

        Returns 
        -------
        idx_sorted : array
            Array of indices that sort the points according to the dictionary 
            order of the 3d subvolumes. 

        slice_array : array 
            array of slice objects used to access the elements of x, y, and z 
            of points residing in a given subvolume. 

        Notes 
        -----
        The dictionary ordering of 3d cells where :math:`dL = L_{\\rm box} / 2` 
        is defined as follows:

            * (0, 0, 0) <--> 0

            * (0, 0, 1) <--> 1

            * (0, 1, 0) <--> 2

            * (0, 1, 1) <--> 3

        And so forth. Each of the Npts thus has a unique triplet, 
        or equivalently, unique integer specifying the subvolume containing the point. 
        The unique integer is called the *cellID*. 
        In order to access the *x* positions of the points lying in subvolume *i*, 
        x[idx_sort][slice_array[i]]. 

        In practice, because fancy indexing with `idx_sort` is not instantaneous, 
        it will be more efficient to use `idx_sort` once to sort the x, y, and z arrays 
        in-place, and then access the sorted arrays with the relevant slice_array element. 
        This is the strategy used in the `retrieve_tree` method. 

        """

        ix = np.floor(x/self.dL[0]).astype(int)
        iy = np.floor(y/self.dL[1]).astype(int)
        iz = np.floor(z/self.dL[2]).astype(int)
        
        #take care of points right on the boundary
        inds = np.where(ix>=self.num_divs[0])[0]
        ix[inds]= self.num_divs[0]-1
        inds = np.where(iy>=self.num_divs[1])[0]
        iy[inds]= self.num_divs[1]-1
        inds = np.where(iz>=self.num_divs[2])[0]
        iz[inds]= self.num_divs[2]-1

        particle_indices = np.ravel_multi_index((ix, iy, iz),\
                                               (self.num_divs[0],\
                                                self.num_divs[1],\
                                                self.num_divs[2]))
        
        idx_sorted = np.argsort(particle_indices)
        bin_indices = np.searchsorted(particle_indices[idx_sorted], 
                                      np.arange(np.prod(self.num_divs)))
        bin_indices = np.append(bin_indices, None)
        
        slice_array = np.empty(np.prod(self.num_divs), dtype=object)
        for icell in range(np.prod(self.num_divs)):
            slice_array[icell] = slice(bin_indices[icell], bin_indices[icell+1], 1)
            
        return idx_sorted, slice_array
    
    
    def adjacent_cells(self, *args):
        """ 
        Given a subvolume specified by the input arguments,  
        return the up to length-27 array of cellIDs of the neighboring cells. 
        The input subvolume can be specified either by its ix, iy, iz triplet, 
        or by its cellID. 
        Parameters 
        ----------
        ix, iy, iz : int, optional
            Integers specifying the ix, iy, and iz triplet of the subvolume. 
            If ix, iy, and iz are not passed, then ic must be passed. 
        ic : int, optional
            Integer specifying the cellID of the input subvolume
            If ic is not passed, the ix, iy, and iz must be passed. 
        Returns 
        -------
        result : int array
            up to Length-27 array of cellIDs of neighboring subvolumes. 
        Notes 
        -----
        If one argument is passed to `adjacent_cells`, this argument will be 
        interpreted as the cellID of the input subvolume. 
        If three arguments are passed, these will be interpreted as 
        the ix, iy, iz triplet of the input subvolume. 
        """

        ixgen, iygen, izgen = np.unravel_index(np.arange(3**3), (3, 3, 3)) 

        if len(args) >= 3:
            ix, iy, iz = args[0], args[1], args[2]
        elif len(args) == 1:
            ic = args[0]
            ix, iy, iz = np.unravel_index(ic, (self.num_divs[0],\
                                               self.num_divs[1],\
                                               self.num_divs[2]))

        ixgen = (ixgen + ix - 1) % self.num_divs[0]
        iygen = (iygen + iy - 1) % self.num_divs[1]
        izgen = (izgen + iz - 1) % self.num_divs[2]

        return np.unique(np.ravel_multi_index((ixgen, iygen, izgen), 
                                              (self.num_divs[0],\
                                               self.num_divs[1],\
                                               self.num_divs[2])))


