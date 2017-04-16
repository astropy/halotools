from countpairs cimport countpairs

import ctypes
from libc.stdlib cimport malloc, free
import numpy as np

def cy_countpairs(Lbox, bins, points1, points2=None):
	"""Lbox, bins, points1, points2=None"""
	len_points1 = len(points1)

	#allocate memory for C arrays for first set of points
	cdef double *X1
	cdef double *Y1
	cdef double *Z1
	X1 = <double *>malloc(len_points1*sizeof(double))
	for i in range(len_points1):
		X1[i] = points1[i,0]
	Y1 = <double *>malloc(len_points1*sizeof(double))
	for i in range(len_points1):
		Y1[i] = points1[i,1]
	Z1 = <double *>malloc(len_points1*sizeof(double))
	for i in range(len_points1):
		Z1[i] = points1[i,2]

	#allocate memory for C arrays for first set of points
	cdef double *X2
	cdef double *Y2
	cdef double *Z2

	if (points2==None):
		autocorr = 1
		len_points2 = len_points1
		X2 = <double *>malloc(len_points1*sizeof(double))
		for i in range(len_points1):
			X2[i] = points1[i,0]
		Y2 = <double *>malloc(len_points1*sizeof(double))
		for i in range(len_points1):
			Y2[i] = points1[i,1]
		Z2 = <double *>malloc(len_points1*sizeof(double))
		for i in range(len_points1):
			Z2[i] = points1[i,2]
	else:
		autocorr = 0
		len_points2 = len(points2)
		X2 = <double *>malloc(len_points2*sizeof(double))
		for i in range(len_points2):
			X2[i] = points2[i,0]
		Y2 = <double *>malloc(len_points2*sizeof(double))
		for i in range(len_points2):
			Y2[i] = points2[i,1]
		Z2 = <double *>malloc(len_points2*sizeof(double))
		for i in range(len_points2):
			Z2[i] = points2[i,2]
	xmin=0
	ymin=0
	zmin=0
	xmax=Lbox
	ymax=Lbox
	zmax=Lbox
	max_bin=max(bins)
	len_bins=len(bins)
	cdef double *rbins
	rbins = <double *>malloc(len_bins*sizeof(double))
	for i in range(len_bins):
		rbins[i] = bins[i]

	#create list of pair counts
	cdef int* c_paircounts 
	c_paircounts= <int *>malloc(len_bins*sizeof(int))

	countpairs(len_points1, X1, Y1, Z1, len_points2, X2, Y2, Z2, xmin, xmax, ymin, ymax, zmin, zmax, autocorr, max_bin, len_bins, rbins, &c_paircounts)
	#convert c array back into python list

	paircounts =[]
	for i in range(len_bins):
		paircounts.append(c_paircounts[i])
	free(c_paircounts)
	return paircounts






