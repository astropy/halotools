#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 19, 2014
#calculate number of pairs with separations given in bins.

from __future__ import division, print_function
try: from mpi4py import MPI
except ImportError:
    print("mpi4py module not available.  MPI functioality will not work.")
import numpy as np
from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree 

def main():
    '''
    example:
    mpirun -np 4 python mpipairs.py output.dat input1.dat input2.dat
    
    Input files should be formatted as N rows of k columns for N k dimensional points.  
        column headers are ok.  e.g. x y z.  If points have weights attached, these must 
        be in a column with a header label 'w'.  In this case, there will be k+1 columns.
    '''
    import sys
    import os
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    #this first option is for my test files, so only works if you have a copy or make one!
    if len(sys.argv)==1:
        savename = './test/test_data/test_out.dat'
        filename_1 = './test/test_data/test_D.dat'
        filename_2 = './test/test_data/test_R.dat'
        if (not os.path.isfile(filename_1)) | (not os.path.isfile(filename_2)):
            raise ValueError('Please provide vailid filenames for data input.')
        else:
            if rank==0:
                print('Running code with default test data files. Saving output as:', savename)
    elif len(sys.argv)==4:
        savename = sys.argv[1]
        filename_1 = sys.argv[2]
        filename_2 = sys.argv[3]
        print('Running code with user supplied data files. Saving output as:', savename)
    else:
        raise ValueError('Please provide a fielpath to save output and two data files to read.')
    
    from astropy.io import ascii
    from astropy.table import Table
    data_1 = ascii.read(filename_1)
    data_2 = ascii.read(filename_2)
    
    #check for weights
    if 'w' in data_1.keys():
        weights_1 = data_1['w'] 
        data_1.remove_column('w')
    else: weights_1 = None
    if 'w' in data_2.keys():
        weights_2 = data_2['w']
        data_2.remove_column('w') 
    else: weights_2 = None
        
    #convert tables into numpy arrays
    data_1 = np.array([data_1[c] for c in data_1.columns]).T
    data_2 = np.array([data_2[c] for c in data_2.columns]).T
    
    #define radial bins.  This should be made an input at some point.
    bins=np.logspace(-1,1.5,10)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    #DD,RR,DR,bins = wnpairs(data_1, data_2, bins, period=None, weights_1=weights_1, weights_2=weights_2, comm=comm)
    DD = wnpairs(data_1, data_1, bins, period=None, weights_1=weights_1, weights_2=weights_1, comm=comm)
    DR = wnpairs(data_1, randoms, bins, period=None, weights_1=weights_1, weights_2=weights_2, comm=comm)
    RR = wnpairs(randoms, randoms, bins, period=None, weights_1=weights_2, weights_2=weights_2, comm=comm)
    
    if rank==0:
        data = Table([bins[1:], DD, RR, DR], names=['r', 'DD', 'RR', 'DR'])
        ascii.write(data, savename)
    

def npairs(data_1, data_2, bins, period=None, comm=None):
    """
    Calculate the number of pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
    
    comm: mpi Intracommunicator object, optional
    
    returns
    -------
    DD_11: data_1-data_1 pairs (auto correlation)
    DD_22: data_2-data_2 pairs (auto correlation)
    DD_12: data_1-data_2 pairs (cross-correlation)
    bins
    """
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(data_1)[-1]!=np.shape(data_2)[-1]:
        raise ValueError("data_1 and data_2 inputs do not have the same dimension.")
        return None
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data_1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period)[0] != np.shape(data_1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    N1 = len(data_1)
    N2 = len(data_2)
    
    #define the indices
    inds1 = np.arange(0,N1)
    inds2 = np.arange(0,N2)
    
    #split up indices for each subprocess
    sendbuf_1=[] #need these as place holders till each process get its list
    sendbuf_2=[]
    if rank==0:
        chunks = np.array_split(inds1,size) #evenly split up the indices
        sendbuf_1 = chunks
        chunks = np.array_split(inds2,size)
        sendbuf_2 = chunks
    
    if comm!=None:
        #send out lists of indices for each subprocess to use
        inds1=comm.scatter(sendbuf_1,root=0)
        inds2=comm.scatter(sendbuf_2,root=0)
    
    #creating trees seems very cheap, so I don't worry about this too much.
    #create full trees
    #KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    #create chunked up trees
    KDT_1_small = cKDTree(data_1[inds1])
    #KDT_2_small = cKDTree(data_2[inds2])
    
    #count!
    #counts_11 = KDT_1_small.count_neighbors(KDT_1, bins, period=period)
    #counts_22 = KDT_2_small.count_neighbors(KDT_2, bins, period=period)
    counts_12 = KDT_1_small.count_neighbors(KDT_2, bins, period=period)
    #DD_11     = counts_11
    #DD_22     = counts_22
    DD_12     = counts_12
    
    if comm==None:
        #return DD_11, DD_12, DD_22, bins
        return DD_12
    else:
        #gather results from each subprocess
        #DD_11 = comm.gather(DD_11,root=0)
        #DD_22 = comm.gather(DD_22,root=0)
        DD_12 = comm.gather(DD_12,root=0)
        
    if (rank==0) & (comm!=None):
        #combine counts from subprocesses
        #DD_11=np.sum(DD_11, axis=0)
        #DD_22=np.sum(DD_22, axis=0)
        DD_12=np.sum(DD_12, axis=0)
    
    #receive result from rank 0
    #DD_11 = comm.bcast(DD_11, root=0)
    #DD_22 = comm.bcast(DD_22, root=0)
    DD_12 = comm.bcast(DD_12, root=0)
    
    #return DD_11, DD_12, DD_22, bins
    return DD_12


def wnpairs(data_1, data_2, bins, period=None , weights1=None, weights2=None, wf=None, aux1=None, aux2=None, comm=None):
    """
    Calculate the weighted number of pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
        
    weights1: array_like, optional
        length N1 array containing weights used for weighted pair counts, w1*w2.
        
    weights2: array_like, optional
        length N2 array containing weights used for weighted pair counts, w1*w2.
    
    aux1: array_like, optional
        length N1 array containing secondary weights used for weighted pair counts.
        
    aux2: array_like, optional
        length N2 array containing secondary weights used for weighted pair counts.
    
    wf: function object, optional
        weighting function.  default is w(w1,w2) returns w1*w2
    
    comm: mpi Intracommunicator object, optional
    
    returns
    -------
    DD_11: data_1-data_1 weighted pairs (auto correlation)
    DD_22: data_2-data_2 weighted pairs (auto correlation)
    DD_12: data_1-data_2 weighted pairs (cross-correlation)
    bins
    """
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(data_1)[-1]!=np.shape(data_2)[-1]:
        raise ValueError("data_1 and data_2 inputs do not have the same dimension.")
        return None
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data_1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period)[0] != np.shape(data_1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(data_1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data_1)[0]:
            raise ValueError("weights_1 should have same len as data_1")
            return None
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(data_2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data_2)[0]:
            raise ValueError("weights_2 should have same len as data_2")
            return None
    
    #Process aux1 entry and check for consistency.
    if aux1 is None:
            aux1 = np.array([1.0]*np.shape(data_1)[0], dtype=np.float64)
    else:
        aux1 = np.asarray(aux1).astype("float64")
        if np.shape(aux1)[0] != np.shape(data_1)[0]:
            raise ValueError("aux1 should have same len as data_1")
            return None
    #Process aux2 entry and check for consistency.
    if aux2 is None:
            aux2 = np.array([1.0]*np.shape(data_2)[0], dtype=np.float64)
    else:
        aux2 = np.asarray(aux2).astype("float64")
        if np.shape(aux2)[0] != np.shape(data_2)[0]:
            raise ValueError("weights_2 should have same len as data_2")
            return None
    
    N1 = len(data_1)
    N2 = len(data_2)
    
    #define the indices
    inds1 = np.arange(0,N1)
    inds2 = np.arange(0,N2)
    
    #split up indices for each subprocess
    sendbuf_1=[] #need these as place holders till each process get its list
    sendbuf_2=[]
    if rank==0:
        chunks = np.array_split(inds1,size)
        sendbuf_1 = chunks
        chunks = np.array_split(inds2,size)
        sendbuf_2 = chunks
    
    if comm!=None:
        #send out lists of indices for each subprocess to use
        inds1=comm.scatter(sendbuf_1,root=0)
        inds2=comm.scatter(sendbuf_2,root=0)
    
    #creating trees seems very cheap, so I don't worry about this too much.
    #create full trees
    #KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    #create chunked up trees
    KDT_1_small = cKDTree(data_1[inds1])
    #KDT_2_small = cKDTree(data_2[inds2])
    
    #count!
    #counts_11 = KDT_1_small.wcount_neighbors(KDT_1, bins, period=period,\
    #    sweights=weights1[inds1], oweights=weights1, w=wf)
    #counts_22 = KDT_2_small.wcount_neighbors(KDT_2, bins, period=period,\
    #    sweights=weights2[inds2], oweights=weights2, w=wf)
    counts_12 = KDT_1_small.wcount_neighbors(KDT_2, bins, period=period,\
        sweights=weights1[inds1], oweights=weights2, w=wf, saux=aux1[inds1], oaux=aux2)
    #DD_11     = counts_11
    #DD_22     = counts_22
    DD_12     = counts_12
    
    if comm==None:
        #return DD_11, DD_12, DD_22, bins
        return DD_12
    else:
        #gather results from each subprocess
        #DD_11 = comm.gather(DD_11,root=0)
        #DD_22 = comm.gather(DD_22,root=0)
        DD_12 = comm.gather(DD_12,root=0)
        
    if (rank==0) & (comm!=None):
        #combine counts from subprocesses
        #DD_11=np.sum(DD_11, axis=0)
        #DD_22=np.sum(DD_22, axis=0)
        DD_12=np.sum(DD_12, axis=0)
    
    #receive result from rank 0
    #DD_11 = comm.bcast(DD_11, root=0)
    #DD_22 = comm.bcast(DD_22, root=0)
    DD_12 = comm.bcast(DD_12, root=0)
    
    #return DD_11, DD_12, DD_22, bins
    return DD_12


def specific_wnpairs(data_1, data_2, bins, period=None , weights1=None, weights2=None, wf=None, aux1=None, aux2=None, comm=None):
    """
    Calculate the weighted number of pairs with separations less than or equal to rbins[i]
    for each point in data_1.
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
        
    weights1: array_like, optional
        length N1 array containing weights used for weighted pair counts, w1*w2.
        
    weights2: array_like, optional
        length N2 array containing weights used for weighted pair counts, w1*w2.
    
    aux1: array_like, optional
        length N1 array containing secondary weights used for weighted pair counts.
        
    aux2: array_like, optional
        length N2 array containing secondary weights used for weighted pair counts.
    
    wf: function object, optional
        weighting function.  default is w(w1,w2) returns w1*w2
    
    comm: mpi Intracommunicator object, optional
    
    returns
    -------
    DD_12: data_1-data_2 weighted pairs counts
    """
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(data_1)[-1]!=np.shape(data_2)[-1]:
        raise ValueError("data_1 and data_2 inputs do not have the same dimension.")
        return None
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data_1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period)[0] != np.shape(data_1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(data_1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data_1)[0]:
            raise ValueError("weights_1 should have same len as data_1")
            return None
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(data_2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data_2)[0]:
            raise ValueError("weights_2 should have same len as data_2")
            return None
    
    #Process aux1 entry and check for consistency.
    if aux1 is None:
            aux1 = np.array([1.0]*np.shape(data_1)[0], dtype=np.float64)
    else:
        aux1 = np.asarray(aux1).astype("float64")
        if np.shape(aux1)[0] != np.shape(data_1)[0]:
            raise ValueError("aux1 should have same len as data_1")
            return None
    #Process aux2 entry and check for consistency.
    if aux2 is None:
            aux2 = np.array([1.0]*np.shape(data_2)[0], dtype=np.float64)
    else:
        aux2 = np.asarray(aux2).astype("float64")
        if np.shape(aux2)[0] != np.shape(data_2)[0]:
            raise ValueError("weights_2 should have same len as data_2")
            return None
    
    N1 = len(data_1)
    N2 = len(data_2)
    
    #define the indices
    inds1 = np.arange(0,N1)
    inds2 = np.arange(0,N2)
    
    #split up indices for each subprocess
    sendbuf_1=[] #need these as place holders till each process get its list
    sendbuf_2=[]
    if rank==0:
        chunks = np.array_split(inds1,size)
        sendbuf_1 = chunks
        chunks = np.array_split(inds2,size)
        sendbuf_2 = chunks
    
    if comm!=None:
        #send out lists of indices for each subprocess to use
        inds1=comm.scatter(sendbuf_1,root=0)
        inds2=comm.scatter(sendbuf_2,root=0)
    
    #creating trees seems very cheap, so I don't worry about this too much.
    #create full trees
    #KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    #create chunked up trees
    KDT_1_small = cKDTree(data_1[inds1])
    #KDT_2_small = cKDTree(data_2[inds2])
    
    #count!
    counts_12 = KDT_1_small.wcount_neighbors_custom(KDT_2, bins, period=period,\
                            sweights=weights1[inds1], oweights=weights2, w=wf,\
                            saux=aux1[inds1], oaux=aux2)
    DD_12     = counts_12
    
    if comm==None:
        return DD_12
    else:
        #gather results from each subprocess
        DD_12 = comm.gather(DD_12,root=0)
        
    if (rank==0) & (comm!=None):
        #combine counts from subprocesses
        DD_12=np.vstack(DD_12)
    
    #receive result from rank 0
    DD_12 = comm.bcast(DD_12, root=0)
    
    return DD_12


def jnpairs(data_1, data_2, bins, period=None , weights1=None, weights2=None, N_vol_elements=None, comm=None):
    """
    Calculate the jackknife number of pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
        
    weights_1: array_like, optional
        length N1 array containing weights used for weighted pair counts
        
    weights_2: array_like, optional
        length N2 array containing weights used for weighted pair counts.
    
    N_vol_elements: int, optional
        number of volume elements in jackknife sample.  
    
    comm: mpi Intracommunicator object, optional
    
    returns
    -------
    DD_11: data_1-data_1 jackknife weighted pairs (auto correlation)
    DD_22: data_2-data_2 jackknife weighted pairs (auto correlation)
    DD_12: data_1-data_2 jackknife weighted pairs (cross-correlation)
    bins
    
    note: pair counts are returned in (len(rbins),N_vol_elements) shape numpy arrays.
    The first row is the pair counts for the full sample, the remaining i rows are the pair 
    counts in the ith jackknife samples 
    """
    
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(data_1)[-1]!=np.shape(data_2)[-1]:
        raise ValueError("data_1 and data_2 inputs do not have the same dimension.")
        return None
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data_1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period)[0] != np.shape(data_1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(data_1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data_1)[0]:
            raise ValueError("weights_1 should have same len as data_1")
            return None
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(data_2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data_2)[0]:
            raise ValueError("weights_2 should have same len as data_2")
            return None
    
    N1 = len(data_1)
    N2 = len(data_2)
    wdim = N_vol_elements+1
    
    #define the indices
    inds1 = np.arange(0,N1)
    inds2 = np.arange(0,N2)
    
    #split up indices for each subprocess
    sendbuf_1=[] #need these as place holders till each process get its list
    sendbuf_2=[]
    if rank==0:
        chunks = np.array_split(inds1,size)
        sendbuf_1 = chunks
        chunks = np.array_split(inds2,size)
        sendbuf_2 = chunks
    
    if comm!=None:
        #send out lists of indices for each subprocess to use
        inds1=comm.scatter(sendbuf_1,root=0)
        inds2=comm.scatter(sendbuf_2,root=0)
    
    #creating trees seems very cheap, so I don't worry about this too much.
    #create full trees
    #KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    #create chunked up trees
    KDT_1_small = cKDTree(data_1[inds1])
    #KDT_2_small = cKDTree(data_2[inds2])
    
    #count!
    #counts_11 = KDT_1_small.wcount_neighbors_custom_2D(KDT_1, bins, period=period,\
    #    sweights=weights1[inds1], oweights=weights1, wdim=wdim)
    #counts_22 = KDT_2_small.wcount_neighbors_custom_2D(KDT_2, bins, period=period,\
    #    sweights=weights2[inds2], oweights=weights2, wdim=wdim)
    counts_12 = KDT_1_small.wcount_neighbors_custom_2D(KDT_2, bins, period=period,\
        sweights=weights1[inds1], oweights=weights2, wdim=wdim)
    #DD_11     = counts_11
    #DD_22     = counts_22
    DD_12     = counts_12
    
    if comm==None:
        #return DD_11, DD_12, DD_22, bins
        return DD_12
    else:
        #gather results from each subprocess
        #DD_11 = comm.gather(DD_11,root=0)
        #DD_22 = comm.gather(DD_22,root=0)
        DD_12 = comm.gather(DD_12,root=0)
        
    if (rank==0) & (comm!=None):
        #combine counts from subprocesses
        #DD_11=np.sum(DD_11, axis=0)
        #DD_22=np.sum(DD_22, axis=0)
        DD_12=np.sum(DD_12, axis=0)
    
    #receive result from rank 0
    #DD_11 = comm.bcast(DD_11, root=0)
    #DD_22 = comm.bcast(DD_22, root=0)
    DD_12 = comm.bcast(DD_12, root=0)
    
    #return DD_11, DD_12, DD_22, bins
    return DD_12


if __name__ == '__main__':
    main()
    