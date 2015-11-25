# -*- coding: utf-8 -*-

"""
galaxy group classes
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from scipy.sparse import csgraph, csr_matrix, coo_matrix
from math import pi, gamma
from ..custom_exceptions import *
from .pair_counters.double_tree_pair_matrix import *

igraph_available=True
try: import igraph
except ImportError:
    igraph_available=False
if igraph_available==True: #there is another package called igraph--need to distinguish.
    if not hasattr(igraph,'Graph'):
        igraph_available==False
no_igraph_msg = ("igraph package not installed.  Some functions will not be available. \n"
                 "See http://igraph.org/ and note that there are two packages called \n"
                 "'igraph'.")
##########################################################################################

__all__=['FoFGroups']
__author__ = ['Duncan Campbell']

class FoFGroups(object):
    """
    Friends-of-friends (FoF) groups class.
    """
    
    def __init__(self, positions, b_perp, b_para, period=None, Lbox=None, num_threads=1):
        """
        Build FoF groups in redshift space groups assuming the distant observer approximation.
        
        The first two dimensions (x, y) define the plane for perpendicular distances. 
        The third dimension (z) is used for parallel distances.  i.e. x,y positions are 
        on the plane of the sky, and z is the radial distance coordinate.  This is the 
        'distant observer' approximation.
    
        Parameters
        ----------
        positions : array_like
            Npts x 3 numpy array containing 3-D positions of galaxies. 
        
        b_perp : float
            Normalized maximum linking length in the perpendicular direction.
            Normalized to the mean separation between galaxies. 
        
        b_para : float
            Normalized maximum linking length in the parallel direction. 
            Normalized to the mean separation between galaxies. 
        
        period : array_like, optional
            length 3 array defining axis-aligned periodic boundary conditions.
        
        Lbox : array_like, optional
            length 3 array defining cuboid boundaries of the simulation box.
        
        num_threads : int, optional
           number of threads to use in calculation. Default is 1. A string 'max' may be 
           used to indicate that the pair counters should use all available cores on
           the machine.
        
        Examples
        --------
        >>> #randomly distributed points in a unit cube. 
        >>> Npts = 1000
        >>> x,y,z = (np.random.random(Npts),np.random.random(Npts),np.random.random(Npts))
        >>> coords = np.vstack((x,y,z)).T
        >>> period = np.array([1.0,1.0,1.0])
        >>> groups = FoFGroups(coords, 0.2, 0.1, period=period)
        
        """
        
        self.b_perp = float(b_perp) #perpendicular linking length
        self.b_para = float(b_para) #parallel linking length
        self.positions=np.asarray(positions,dtype=np.float64) #coordinates of galaxies
        
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
    
        self.period = period #simulation box periodic boundary conditions
        self.Lbox = np.asarray(Lbox,dtype='float64') #simulation box periodic boundary conditions
    
        #calculate the physical linking lengths
        self.volume = np.prod(self.Lbox)
        self.n_gal = len(positions)/self.volume
        self.d_perp = self.b_perp/(self.n_gal**(1.0/3.0))
        self.d_para = self.b_para/(self.n_gal**(1.0/3.0))
    
        self.m_perp, self.m_para = xy_z_pair_matrix(self.positions, self.positions,\
                                                  self.d_perp, self.d_para,\
                                                  period=self.period, Lbox=self.Lbox,\
                                                  num_threads=num_threads)
        
        self.m = self.m_perp.multiply(self.m_perp)+self.m_para.multiply(self.m_para)
        self.m = self.m.sqrt()
    
    @property
    def group_ids(self):
        """
        Determine integer IDs for groups.
        
        Each member of a group is assigned a unique integer ID that it shares with all 
        connected group members.
        
        Returns
        -------
        group_ids : np.array
            array of group IDs for each galaxy
        """
        if getattr(self,'_group_ids',None) is None:
            self._n_groups, self._group_ids = csgraph.connected_components(\
                                                  self.m_perp, directed=False,\
                                                  return_labels=True)
        return self._group_ids
    
    @property
    def n_groups(self):
        """
        Calculate the total number of groups, including 1 member groups
        
        Returns
        -------
        N_groups: int
            number of distinct groups
        """
        if getattr(self,'_n_groups',None) is None:
            self._n_groups = csgraph.connected_components(self.m_perp, directed=False,\
                                                          return_labels=False)
        return self._n_groups
    
    ####the following methods are igraph package dependent###
    def create_graph(self):
        """
        Create graph from FoF sparse matrix (requires igraph package).
        """
        if igraph_available==True:
            self.g = _scipy_to_igraph(self.m, self.positions, directed=False)
        else: raise HalotoolsError(no_igraph_msg)
    
    def get_degree(self):
        """
        Calculate the 'degree' of each galaxy vertex (requires igraph package).
        
        Returns
        -------
        degree : np.array
            the 'degree' of galaxies in groups
        """
        if igraph_available==True:
            self.degree = self.g.degree()
            return self.degree
        else: raise HalotoolsError(no_igraph_msg)
    
    def get_betweenness(self):
        """
        Calculate the 'betweenness' of each galaxy vertex (requires igraph package).
        
        Returns
        -------
        betweeness : np.array
            the 'betweenness' of galaxies in groups
        """
        if igraph_available==True:
            self.betweenness = self.g.betweenness()
            return self.betweenness
        else: raise HalotoolsError(no_igraph_msg)
    
    def get_multiplicity(self):
        """
        Return the multiplicity of galaxies' group (requires igraph package).
        """
        if igraph_available==True:
            clusters = self.g.clusters()
            mltp = np.array(clusters.sizes())
            self.multiplicity = mltp[self.group_ids]
            return self.multiplicity
        else: raise HalotoolsError(no_igraph_msg)
    
    def get_edges(self):
        """
        Return all edges of the graph (requires igraph package).
        
        Returns
        -------
        edges: np.ndarray
            N_edges x 2 array of vertices that are connected by an edge.  The vertices are
            indicated by their index.
        """
        if igraph_available==True:
            self.edges = np.asarray(self.g.get_edgelist())
            return self.edges
        else: raise HalotoolsError(no_igraph_msg)
    
    def get_edge_lengths(self):
        """
        return the length of all edges (requires igraph package).
        
        Returns
        -------
        lengths: np.array
            The length of an 'edge' econnnecting galaxies, i.e. distance between galaxies.
        
        Notes
        ------
        The length is caclulated as
        .. math::
            L_{\edge} = \\sqrt(r_{\\perp}^2 + r_{\\parallel}^2},
        
        where :math:`r_{\\perp}` and :math:`r_{\\parallel}` are the perendicular an 
        parallel distance between galaixes.
        """
        if igraph_available==True:
            edges = self.g.es()
            lens = edges.get_attribute_values('weight')
            self.edge_lengths = np.array(lens)
            return self.edge_lengths
        else: raise HalotoolsError(no_igraph_msg)


def _scipy_to_igraph(matrix, coords, directed=False):
    """
    Convert a scipy sparse matrix to an igraph graph object (requires igraph package).
    
    Paramaters
    ----------
    matrix : object
        scipy.sparse pairwise distance matrix
    
    coords : np.array
        N by 3 array of coordinates of points
        
    Returns
    -------
    graph : object
        igraph graph object
    """
    
    matrix = csr_matrix(matrix)
    sources, targets = matrix.nonzero()
    weights = matrix[sources, targets].tolist()[0]
    
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    if igraph_available:
        g = igraph.Graph(zip(sources, targets), n=matrix.shape[0], directed=directed,\
                            edge_attrs={'weight': weights},\
                            vertex_attrs={'x':x, 'y':y, 'z':z })
        return g
    else:  raise HalotoolsError(no_igraph_msg)
    
    
    
    