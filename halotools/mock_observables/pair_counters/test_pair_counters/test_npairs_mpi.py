#!/usr/bin/env python

#import modules
from __future__ import division, print_function
import numpy as np
import sys

from ..mpipairs import npairs, wnpairs, jnpairs

try: 
    from mpi4py import MPI
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False

from ..mpipairs import npairs, wnpairs, jnpairs
#import simple pair counter to compare results
from ..pairs import npairs as comp_npairs
from ..pairs import wnpairs as comp_wnpairs

__all__ = ['test_npairs', 'test_npairs_speed', 'test_wnpairs', 'test_wnpairs_speed',\
           'test_jnpairs', 'test_jnpairs_speed']

"""
This script tests the functionality of npairs_mpi.py
"""

def main():
    """
    use this main function to test functions with mpirun.
    e.g. mpirun -np 4 test_npairs_mpi.py
    """
    
    if len(sys.argv)>1: N_points = int(sys.argv[1])
    else: N_points = 100
    
    import time
    
    #initialize communication object
    if mpi4py_installed==True:
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        comm = None
        rank=0
    
    if rank==0: start = time.time()
    test_npairs_speed(comm=comm, N1=N_points, N2=N_points)
    if rank==0:
        dt = time.time()-start
        print('npairs ran in: {0} seconds'.format(dt))
    if rank==0: start = time.time()
    test_wnpairs_speed(comm=comm, N1=N_points, N2=N_points)
    if rank==0:
        dt = time.time()-start
        print('wnpairs ran in: {0} seconds'.format(dt))
    if rank==0: start = time.time()
    test_jnpairs_speed(comm=comm, N1=N_points, N2=N_points)
    if rank==0:
        dt = time.time()-start
        print('jnpairs ran in: {0} seconds'.format(dt))


def test_npairs(comm=None, N1=100, N2=100):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()

    if rank==0:
        #create some dummy random data
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
    
    D1D2 = npairs(sample1, sample2, rbins, period=period, comm=comm)

    D1D2_comp = comp_npairs(sample1, sample2, rbins, period=period)
        
    assert np.all(D1D2==D1D2_comp), "mpi pair counts do not match simple pair counter"


def test_npairs_speed(comm=None, N1=100, N2=100):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()

    if rank==0:
        #create some dummy random data
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
    
    D1D2 = npairs(sample1, sample2, rbins, period=period, comm=comm)

    pass


def test_wnpairs(comm=None, N1=100, N2=100):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()

    #create some dummy random data
    if rank>=0:
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

    D1D1 = wnpairs(sample1, sample1, rbins, period=period,\
                                  weights1=weights1, weights2=weights1, comm=comm)

    D1D1_comp = comp_wnpairs(sample1, sample1, rbins, period=period,\
                             weights1=weights1, weights2=weights1)
    
    epsilon = np.float64(sys.float_info[8])

    assert np.all(np.nan_to_num(np.fabs(D1D1-D1D1_comp)/D1D1)<100*epsilon),\
        "mpi pair weighted counts do not match simple weighted pair counter" 


def test_wnpairs_speed(comm=None, N1=100, N2=100):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()

    #create some dummy random data
    if rank>=0:
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

    D1D1 = wnpairs(sample1, sample1, rbins, period=period,\
                                  weights1=weights1, weights2=weights1, comm=comm)

    pass


def test_jnpairs(comm=None, N1=100, N2=100):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    N_sub_vol = 10 #number of jackknife samples
    #create some dummy random data
    if rank==0:
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
    
    D1D2 = jnpairs(sample1, sample2, rbins, period=period,\
                                  weights1=weights1, weights2=weights2,\
                                  N_vol_elements=N_sub_vol, comm=comm)
    
    D1D2_comp = comp_npairs(sample1, sample2, rbins, period=period)
                      
    assert np.all(D1D2[0]==D1D2_comp),\
        "mpi pair jackknife full sample counts do not match simple pair counter"


def test_jnpairs_speed(comm=None, N1=100, N2=100):
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    N_sub_vol = 10 #number of jackknife samples
    #create some dummy random data
    if rank==0:
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
    
    D1D2 = jnpairs(sample1, sample2, rbins, period=period,\
                                  weights1=weights1, weights2=weights2,\
                                  N_vol_elements=N_sub_vol, comm=comm)
    
    pass
    

if __name__ == '__main__':
    main()
