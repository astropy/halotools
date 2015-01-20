#!/usr/bin/env python

#Duncan Campbell
#Yale University
#October 6, 2014
#test out the Delta Sigma code.


from __future__ import division, print_function
import numpy as np
import halotools.make_mocks
from halotools.mock_observables.observables import Delta_Sigma

def main():
    mock = halotools.make_mocks.HOD_mock()
    mock.populate()
    
    #galaxy_selection = np.random.permutation(np.arange(0,len(mock.galaxies)))
    #galaxy_selection = galaxy_selection[0:100]
    
    galaxy_selection = np.where((mock.galaxies['primary_halo_property']>14.7) & (mock.galaxies['isSat']==1))[0]
    print(len(galaxy_selection))
    
    rbins = np.zeros((26,))
    rbins[1:] = np.logspace(np.log10(0.2),np.log10(2),25)
    
    print(rbins)
    
    result = Delta_Sigma(mock.galaxies['coords'][galaxy_selection],\
                         mock.particles['POS'],\
                         rbins, bounds=[-50.0,50.0], period = mock.Lbox)
                
    print(result)

if __name__ == '__main__':
    main()