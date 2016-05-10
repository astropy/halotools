""" Module providing testing for 
`~halotools.mock_observables.conditional_spherical_isolation` function. 
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import pytest 
from astropy.utils.misc import NumpyRNGContext

from ..spherical_isolation import spherical_isolation
from ..conditional_spherical_isolation import conditional_spherical_isolation

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ('test_conditional_spherical_isolation_cond_func1', )

@pytest.mark.xfail
def test_conditional_spherical_isolation_args_processing1():
    npts = 100
    sample1 = np.random.random((npts, 3))
    r_max = 0.1
    iso = conditional_spherical_isolation(sample1, sample1, r_max, period=1)


def test_conditional_spherical_isolation_cond_func1():
    sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
    sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
    r_max = 0.2
    marks1 = np.ones(len(sample1))
    marks2 = np.zeros(len(sample2))
    
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == False)

    cond_func = 1
    marked_iso1a = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks1, marks2, cond_func, period=1)
    marked_iso1b = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks2, marks1, cond_func, period=1)
    assert np.all(marked_iso1a == False)
    assert np.all(marked_iso1b == True)

def test_conditional_spherical_isolation_cond_func2():
    sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
    sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
    r_max = 0.2
    marks1 = np.ones(len(sample1))
    marks2 = np.zeros(len(sample2))
    
    iso = spherical_isolation(sample1, sample2, r_max, period=1.0)
    assert np.all(iso == False)

    cond_func = 2
    marked_iso2a = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks1, marks2, cond_func, period=1)
    marked_iso2b = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks2, marks1, cond_func, period=1)
    assert np.all(marked_iso2a == True)
    assert np.all(marked_iso2b == False)

def test_conditional_spherical_isolation_cond_func3():
    sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
    sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
    r_max = 0.2
    marks1 = np.ones(len(sample1))
    marks2 = np.zeros(len(sample2))
    
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == False)

    cond_func = 3
    marked_iso3 = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks1, marks2, cond_func, period=1)
    assert np.all(marked_iso3 == True)

def test_conditional_spherical_isolation_cond_func4():
    sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
    sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
    r_max = 0.2
    marks1 = np.ones(len(sample1))
    marks2 = np.zeros(len(sample2))
    
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == False)

    cond_func = 4
    marked_iso4 = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks1, marks1, cond_func, period=1)
    assert np.all(marked_iso4 == True)

def test_conditional_spherical_isolation_cond_func5():
    sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
    sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
    r_max = 0.25

    # First verify that the two points are not isolated when ignoring the marks
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == False)

    # The first sample1 mark is 1, the second sample1 mark is also 1
    # All sample2 marks are 0
    # Thus w_1[0] > (w_2[0]+w_1[1]) NEVER holds, 
    # and so the marked isolation should always be True
    marks1a, marks1b = np.ones(len(sample1)), np.ones(len(sample1)) 
    marks1 = np.vstack([marks1a, marks1b]).T 
    marks2a, marks2b = np.zeros(len(sample2)), np.zeros(len(sample2)) 
    marks2 = np.vstack([marks2a, marks2b]).T 
    
    cond_func = 5
    marked_iso5 = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks1, marks2, cond_func, period=1)
    assert np.all(marked_iso5 == True)

    # The first sample1 mark is 1, the second sample1 mark is 0
    # All sample2 marks are 0
    # Thus w_1[0] > (w_2[0]+w_1[1]) ALWAYS holds, 
    # and so the marked isolation should be equivalent to the unmarked isolation 
    marks1a, marks1b = np.ones(len(sample1)), np.zeros(len(sample1)) 
    marks1 = np.vstack([marks1a, marks1b]).T 
    marks2a, marks2b = np.zeros(len(sample2)), np.zeros(len(sample2)) 
    marks2 = np.vstack([marks2a, marks2b]).T 
    
    cond_func = 5
    marked_iso5 = conditional_spherical_isolation(sample1, sample2, r_max, 
        marks1, marks2, cond_func, period=1)
    assert np.all(marked_iso5 == False)

def test_stellar_mass_conditional_spherical_isolation_correctness_cond_func1():
    """ Create a regular 3d grid for sample2 and create a single point 
    for sample1 that lies in the center of one of the grid cells. 

    Set the ``stellar_mass`` of the points in the grid to be 2e10 and the 
    ``stellar_mass`` of the single sample1 point to be 1e10. Verify that 
    the conditional isolation functions behave properly in all  
    limits of the weighting functions. 

    Testing function for conf_func_id = 1
    """
    sample1 = np.zeros((1, 3)) + 0.2
    sample2 = generate_3d_regular_mesh(5) # grid-coords = 0.1, 0.3, 0.5, 0.7, 0.9
    stellar_mass2 = np.zeros(len(sample2)) + 2e10

    # First verify that unconditioned spherical isolation behaves properly for this sample
    r_max = 0.05
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == False)

    cond_func = 1

    r_max = 0.05
    stellar_mass1 = np.zeros(len(sample1)) + 1e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 1e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 3e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == False)

def test_stellar_mass_conditional_spherical_isolation_correctness_cond_func2():
    """ Create a regular 3d grid for sample2 and create a single point 
    for sample1 that lies in the center of one of the grid cells. 

    Set the ``stellar_mass`` of the points in the grid to be 2e10 and the 
    ``stellar_mass`` of the single sample1 point to be 1e10. Verify that 
    the conditional isolation functions behave properly in all  
    limits of the weighting functions. 

    Testing function for conf_func_id = 2
    """
    sample1 = np.zeros((1, 3)) + 0.2
    sample2 = generate_3d_regular_mesh(5) # grid-coords = 0.1, 0.3, 0.5, 0.7, 0.9
    stellar_mass2 = np.zeros(len(sample2)) + 2e10

    # First verify that unconditioned spherical isolation behaves properly for this sample
    r_max = 0.05
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == False)

    cond_func = 2

    r_max = 0.05
    stellar_mass1 = np.zeros(len(sample1)) + 1e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 1e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == False)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 3e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

def test_stellar_mass_conditional_spherical_isolation_correctness_cond_func3():
    """ Create a regular 3d grid for sample2 and create a single point 
    for sample1 that lies in the center of one of the grid cells. 

    Set the ``stellar_mass`` of the points in the grid to be 2e10 and the 
    ``stellar_mass`` of the single sample1 point to be 1e10. Verify that 
    the conditional isolation functions behave properly in all  
    limits of the weighting functions. 

    Testing function for conf_func_id = 3
    """
    sample1 = np.zeros((1, 3)) + 0.2
    sample2 = generate_3d_regular_mesh(5) # grid-coords = 0.1, 0.3, 0.5, 0.7, 0.9
    stellar_mass2 = np.zeros(len(sample2)) + 2e10

    # First verify that unconditioned spherical isolation behaves properly for this sample
    r_max = 0.05
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == False)

    cond_func = 3

    r_max = 0.05
    stellar_mass1 = np.zeros(len(sample1)) + 2e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 3e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 2e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == False)


def test_stellar_mass_conditional_spherical_isolation_correctness_cond_func4():
    """ Create a regular 3d grid for sample2 and create a single point 
    for sample1 that lies in the center of one of the grid cells. 

    Set the ``stellar_mass`` of the points in the grid to be 2e10 and the 
    ``stellar_mass`` of the single sample1 point to be 1e10. Verify that 
    the conditional isolation functions behave properly in all  
    limits of the weighting functions. 

    Testing function for conf_func_id = 4
    """
    sample1 = np.zeros((1, 3)) + 0.2
    sample2 = generate_3d_regular_mesh(5) # grid-coords = 0.1, 0.3, 0.5, 0.7, 0.9
    stellar_mass2 = np.zeros(len(sample2)) + 2e10

    # First verify that unconditioned spherical isolation behaves properly for this sample
    r_max = 0.05
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == False)

    cond_func = 4

    r_max = 0.05
    stellar_mass1 = np.zeros(len(sample1)) + 2e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 3e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == False)

    r_max = 0.2
    stellar_mass1 = np.zeros(len(sample1)) + 2e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

def test_stellar_mass_conditional_spherical_isolation_correctness_cond_func5():
    """ Create a regular 3d grid for sample2 and create a single point 
    for sample1 that lies in the center of one of the grid cells. 

    Set the ``stellar_mass`` of the points in the grid to be 2e10 and the 
    ``stellar_mass`` of the single sample1 point to be 1e10. Verify that 
    the conditional isolation functions behave properly in all  
    limits of the weighting functions. 

    Testing function for conf_func_id = 5
    """
    sample1 = np.zeros((1, 3)) + 0.2
    sample2 = generate_3d_regular_mesh(5) # grid-coords = 0.1, 0.3, 0.5, 0.7, 0.9
    stellar_mass2 = np.zeros((len(sample2), 2))
    stellar_mass2[:,0] = 2e10

    # First verify that unconditioned spherical isolation behaves properly for this sample
    r_max = 0.05
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert len(iso) == 1
    assert np.all(iso == False)

    cond_func = 5

    r_max = 0.05
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 1e10
    stellar_mass1[:,1] = 1
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 1e10
    stellar_mass1[:,1] = 1
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.05
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 1e10
    stellar_mass1[:,1] = 2e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 1e10
    stellar_mass1[:,1] = 2e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.05
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 3e10
    stellar_mass1[:,1] = 2
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 3e10
    stellar_mass1[:,1] = 2
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == False)

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 2e10
    stellar_mass1[:,1] = 0.
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 3e10
    stellar_mass1[:,1] = 0
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == False)

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 3e10
    stellar_mass1[:,1] = 3e10
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

def test_stellar_mass_conditional_spherical_isolation_correctness_cond_func6():
    """ Create a regular 3d grid for sample2 and create a single point 
    for sample1 that lies in the center of one of the grid cells. 

    Set the ``stellar_mass`` of the points in the grid to be 2e10 and the 
    ``stellar_mass`` of the single sample1 point to be 1e10. Verify that 
    the conditional isolation functions behave properly in all  
    limits of the weighting functions. 

    Testing function for conf_func_id = 6
    """
    sample1 = np.zeros((1, 3)) + 0.2
    sample2 = generate_3d_regular_mesh(5) # grid-coords = 0.1, 0.3, 0.5, 0.7, 0.9
    stellar_mass2 = np.zeros((len(sample2), 2))
    stellar_mass2[:,0] = 2e10

    cond_func = 6

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 3e10
    stellar_mass1[:,1] = 2
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    stellar_mass1 = np.zeros((len(sample1), 2))
    stellar_mass1[:,0] = 1e10
    stellar_mass1[:,1] = 2
    iso = conditional_spherical_isolation(sample1, sample2, 
        r_max, stellar_mass1, stellar_mass2, cond_func, period=1)
    assert np.all(iso == False)









