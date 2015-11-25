#!/usr/bin/env python

#import packages
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import sys
from scipy.sparse import coo_matrix
import pytest 

igraph_available=True
try: import igraph
except ImportError:
    igraph_available=False
    print("igraph package not installed.  Some functions will not be available.")

from ..groups import FoFGroups

__all__=['test_fof_groups_init','test_fof_group_IDs','test_igraph_functionality']

#set random seed to get consistent behavior
np.random.seed(1)
N=1e3
Lbox = np.array([1.0,1.0,1.0])
period = Lbox
sample = np.random.random((N,3))
b_perp = 0.5
b_para = 0.5

@pytest.mark.slow
def test_fof_groups_init():
    """
    test fof_groups object initialization
    """
    
    fof_group = FoFGroups(sample, b_perp, b_para, Lbox=Lbox, period=period)
    
    assert isinstance(fof_group.m_perp,coo_matrix)
    assert isinstance(fof_group.m_para,coo_matrix)

@pytest.mark.slow
def test_fof_group_IDs():
    """
    test group ID caclulation
    """
    
    fof_group = FoFGroups(sample, b_perp, b_para, Lbox=Lbox, period=period)
    
    group_IDs = fof_group.group_ids
    assert len(group_IDs)==N, "number of labels returned is incorrect"
    
    groups = np.unique(group_IDs)
    N_groups = len(groups)
    
    assert N_groups==fof_group.n_groups, "number of groups is incorrect"


@pytest.mark.slow
def test_igraph_functionality():
    """
    test igraph functionality caclulation
    """
    
    fof_group = FoFGroups(sample, b_perp, b_para, Lbox=Lbox, period=period)
    
    if igraph_available==True:
        
        fof_group.create_graph()
        assert isinstance(fof_group.g,igraph.Graph)
        
        degree = fof_group.get_degree()
        assert len(degree) == N
        
        betweenness = fof_group.get_betweenness()
        assert len(betweenness) == N
        
        multiplicity = fof_group.get_multiplicity()
        assert len(multiplicity) == N
    
        edges = fof_group.get_edges()
        #the number of edges is half the sum of vertex degrees of the graph
        assert len(edges)==np.sum(fof_group.degree)/2
        
        lens = fof_group.get_edge_lengths()
        assert len(lens)==len(edges)
        assert np.all(np.sort(lens)==np.sort(fof_group.m.data))
        
    else: pass
    
    
    
