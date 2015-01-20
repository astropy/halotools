#Duncan Campbell
#August 27, 2014
#Yale University

""" 
objects and functions that apply isolation criteria to galaxies in a mock catalog 
"""

from __future__ import division
import sys

__all__=['isolatoion_criterion']

####import modules########################################################################
import numpy as np
from math import pi, gamma
from halotools.mock_observables.spatial import geometry
##########################################################################################

class isolatoion_criterion(object):
    """
    A object that defines a galaxy isolation criterion.
    
    Parameters 
    ----------
    volume: geometry volume object
        e.g. sphere, cylinder
    
    vol_args: list or function
        arguments to initialize the volume objects defining the test region of isolated 
        candidates, or function taking a galaxy object which returns the vol arguments.
    
    test_prop: string
        mock property to test isolation against.  e.g. 'M_r', 'Mstar', etc.
        
    test_func: function
        python function defining the property isolation test.
    """
    
    def __init__(self, volume=geometry.sphere, vol_args=None,
                 test_prop='primary_galprop', test_func=None):
        #check to make sure the volume object passed is in fact a volume object 
        if not issubclass(volume,geometry.volume):
            raise ValueError('volume object must be a subclass of geometry.volume')
        else: self.volume = volume
        #check volume object arguments. Is it None, a function, or a list?
        if vol_args==None:
            #default only passes center argument to volume object
            def default_func(galaxy):
                center = galaxy['coords']
                return center
            self.vol_agrs = default_func
        elif hasattr(vol_args, '__call__'):
            self.vol_args= vol_args
            #check for compatibility with the mock in the method
        else:
            #else, return the list of values passes in every time.
            def default_func(galaxy):
                return vol_agrs
            self.vol_agrs = default_func
        #store these two and check if they are compatible with a mock later in the method.
        self.test_prop = test_prop
        self.test_func = test_func
    
    def make_volumes(self, galaxies, isolated_candidates):
        volumes = np.empty((len(isolated_candidates),))
        for i in range(0,len(isolated_candidates)):
            volumes[i] = self.volume(self.vol_args(galaxies[isolated_candidates[i]]))
        return volumes

    def apply_criterion(self, mock, isolated_candidates):
        """
        Return galaxies which pass isolation criterion. 
    
        Parameters 
        ----------
        mock: galaxy mock object
    
        isolated_candidates: array_like
            indices of mock galaxy candidates to test for isolation.
        
        Returns 
        -------
        inds: numpy.array
            indicies of galaxies in mock that pass the isolation criterion.

        """
        
        #check input
        if not hasattr(mock, 'galaxies'):
            raise ValueError('mock must contain galaxies. execute mock.populate()')
        if self.test_prop not in mock.galaxies.dtype.names:
            raise ValueError('test_prop not present in mock.galaxies table.')
        try: self.volume(self.vol_args(mock.galaxies[0]))
        except TypeError: print('vol_args are not compatable with the volume object.')
        
        volumes = make_volumes(self,mock.galaxies,isolated_candidates)
        
        points_inside_shapes = geometry.inside_volume(
                               volumes, mock.coords[neighbor_candidates], period=mock.Lbox
                               )[2]
        
        ioslated = np.array([True]*len(isolated_candidates))
        for i in range(0,len(isolated_candidates)):
            inside = points_inside_shapes[i] 
            isolated[i] = np.all(self.test_func(mock.galaxies[isolated_candidates[i]][self.test_prop],mock.galaxies[inside][self.test_prop]))
        
        return isolated


