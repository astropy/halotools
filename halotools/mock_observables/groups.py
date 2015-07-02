# -*- coding: utf-8 -*-

"""
functions to create galaxy groups
"""

from __future__ import division, print_function
####import modules########################################################################
import sys
import numpy as np
from scipy.sparse import csgraph
from math import pi, gamma
from .pair_counters.FoF_pairs import fof_pairs, xy_z_fof_pairs
##########################################################################################

__all__=['fof_groups']
__author__ = ['Duncan Campbell']

class fof_groups():
    """
    create friends-of-friends groups object.
    
    The first two dimensions define the plane for perpendicular distances.  The third 
    dimension is used for parallel distances.  i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate.  This is the distant observer approximation.
    
    Parameters
    ----------
    positions : array_like
        Npts x 3 numpy array containing 3-d positions of Npts. 
    
    b_perp : float
        normalized maximum linking length in the perpendicular direction. 
    
    b_para : float
        normalized maximum linking length in the parallel direction. 
    
    period: array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions.
    
    Lbox: array_like, optional
        length 3 array defining cuboid boundaries of the simulation box.
    
    N_threads: int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    Returns
    -------
    labels: np.array
        group_ID labels for each galaxy
    """
    
    def __init__(self, positions, b_perp, b_para, period=None, Lbox=None, N_threads=1):
        
        self.b_perp = b_perp
        self.b_para = b_para
        self.positions=np.asarray(positions)
        
        #process Lbox parameter
        if (Lbox is None) & (period is None): 
            raise ValueError("Lbox and Period cannot be both be None.")
        elif (Lbox is None) & (period is not None):
            Lbox = period
        elif np.shape(Lbox)==():
            Lbox = np.array([Lbox]*3)
        elif np.shape(Lbox)==(1,):
            Lbox = np.array([Lbox[0]]*3)
        else: Lbox = np.array(Lbox)
        if np.shape(Lbox) != (3,):
            raise ValueError("Lbox must be an array of length 3, or number indicating the\
                              length of one side of a cube")
        if (period is not None) and (not np.all(Lbox==period)):
            raise ValueError("If both Lbox and Period are defined, they must be equal.")
    
        self.period = period
        self.Lbox = Lbox
    
        #calculate the physical linking lengths
        self.volume = np.prod(self.Lbox)
        self.n_gal = len(positions)/self.volume
        self.d_perp = self.b_perp/(self.n_gal**(1.0/3.0))
        self.d_para = self.b_para/(self.n_gal**(1.0/3.0))
    
        self.m_perp. self.m_para = xy_z_fof_pairs(self.positions, self.positions, self.d_perp, self.d_para,\
                                period=self.period, Lbox=self.Lbox, N_threads=N_threads)
    
    def get_group_IDs(self):
        n_groups, group_ids = csgraph.connected_components(self.m, directed=False,\
                                                           return_labels=True)
        
        self.n_groups = n_groups
        self.group_ids = group_ids
        
        return self.group_ids
    
    def get_n_groups(self):
        
        self.n_groups = csgraph.connected_components(self.m, directed=False,\
                                                    return_labels=False)
        
        return self.n_groups
    
    