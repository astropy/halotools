#!/usr/bin/env python

#import modules
from __future__ import division, print_function
import numpy as np
import sys
from mpi4py import MPI
from halotools.mock_observables.npairs_mpi import npairs, wnpairs, jnpairs
#import simple pair counter to compare results
from halotools.mock_observables.pairs import npairs as comp_npairs
from halotools.mock_observables.pairs import wnpairs as comp_wnpairs

"""
This script tests the functionality of npairs_mpi.py
"""

def main():
    """
    use this main function to test functions with mpirun.
    e.g. mpirun -np 4 test_npairs_mpi.py
    """
    
    #initialize communication object
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    test_npairs(comm=comm)
    test_wnpairs(comm=comm)
    test_jnpairs(comm=comm)


def test_npairs(comm=None):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()

    if rank==0:
        #create some dummy random data
        N1=100
        N2=100
        sample1 = np.random.random((N1,3))
        sample2 = np.random.random((N2,3))
    else:
        sample1=None
        sample2=None
    
    if comm!=None:
        #make sure each process is using the same data
        sample1 = comm.bcast(sample1, root=0)
        sample2 = comm.bcast(sample2, root=0)
    
    #define PBCs and radial bins
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)
    
    D1D1,D1D2,D2D2,bins = npairs(sample1, sample2, rbins, period=period, comm=comm)

    D1D2_comp = comp_npairs(sample1, sample2, rbins, period=period)
        
    assert np.all(D1D2==D1D2_comp), "mpi pair counts do not match simple pair counter"


def test_wnpairs(comm=None):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()

    #create some dummy random data
    if rank>=0:
        N1=100
        N2=100
        sample1 = np.random.random((N1,3))
        sample2 = np.random.random((N2,3))
        weights1 = np.random.random((N1))
        weights2 = np.random.random((N2))
    else:
        sample1=None
        sample2=None
        weights1=None
        weights2=None

    if comm!=None:
        #make sure each process is using the same data
        sample1 = comm.bcast(sample1, root=0)
        sample2 = comm.bcast(sample2, root=0)
        weights1 = comm.bcast(weights1, root=0)
        weights2 = comm.bcast(weights2, root=0)
    
    #define PBCs and radial bins
    period = np.array([1,1,1])
    rbins = np.linspace(0,0.5,5)

    D1D1,D1D2,D2D2,bins = wnpairs(sample1, sample1, rbins, period=period,\
                                  weights1=weights1, weights2=weights2, comm=comm)

    D1D1_comp = comp_wnpairs(sample1, sample1, rbins, period=period,\
                             weights1=weights1, weights2=weights1)
    
    epsilon = np.float64(sys.float_info[8])

    assert np.all(np.nan_to_num(np.fabs(D1D1-D1D1_comp)/D1D1)<100*epsilon),\
        "mpi pair weighted counts do not match simple weighted pair counter" 


def test_jnpairs(comm=None):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    N_sub_vol = 10 #number of jackknife samples
    #create some dummy random data
    if rank==0:
        N1=100
        N2=1000
        sample1 = np.random.random((N1,3))
        sample2 = np.random.random((N2,3))
        #define random sample labels to points
        weights1 = np.random.random_integers(0,N_sub_vol-1,size=N1)+1 # '0' label is special
        weights2 = np.random.random_integers(0,N_sub_vol-1,size=N2)+1 # '0' label is special
    else:
        sample1=None
        sample2=None
        weights1=None
        weights2=None
    
    if comm!=None:
        #make sure each process is using the same data
        sample1 = comm.bcast(sample1, root=0)
        sample2 = comm.bcast(sample2, root=0)
        weights1 = comm.bcast(weights1, root=0)
        weights2 = comm.bcast(weights2, root=0)
    
    #define PBCs and radial bins
    period = np.array([1.0,1.0,1.0])
    rbins = np.linspace(0,0.5,5)
    
    D1D1,D1D2,D2D2,bins = jnpairs(sample1, sample2, rbins, period=period,\
                                  weights_1=weights1, weights_2=weights2,\
                                  N_vol_elements=N_sub_vol, comm=comm)
    
    D1D2_comp = comp_npairs(sample1, sample2, rbins, period=period)
    
    print(np.shape(D1D1))                    
    assert np.all(D1D2[0]==D1D2_comp),\
        "mpi pair jackknife full sample counts do not match simple pair counter"  
    

if __name__ == '__main__':
    main()
