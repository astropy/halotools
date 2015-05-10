#!/usr/bin/env python

"""
Test the distance functions in distance.py
"""

from ..distances import euclidean_distance
from ..distances import angular_distance
from ..distances import projected_distance

import numpy as np

__all__=['test_euclidean_distance', 'test_angular_distance', 'test_projected_distance']

def test_euclidean_distance():

    x1 = np.array([0.0,0.0,0.0])
    x2 = np.array([1.0,1.0,1.0])
    x3 = np.array([0.0,0.75,0.0])
    
    period = np.array([1,1,1])
    
    d = euclidean_distance(x1,x2,period=None)
    assert d == np.sqrt(3.0), "incorrect distance measure, w/o PBCs"
    
    d = euclidean_distance(x1,x3,period=period)
    assert d == 0.25, "incorrect distance measure, w/ PBCs"


def test_angular_distance():

    p1 = np.array([0.0,0.0])
    p2 = np.array([0.0,90.0])
    
    da = angular_distance(p1,p2)
    
    assert da==90.0, "incorrect angular distance measure."


def test_projected_distance():

    p1 = np.array([0.0,0.0])
    p2 = np.array([0.0,1.0])
    los = np.array([0.0,1.0])
    
    d_para, d_perp = projected_distance(p1,p2,los)
    
    assert d_para==1, "incorrect parallel distance measure."
    assert d_perp==0, "incorrect parallel distance measure."