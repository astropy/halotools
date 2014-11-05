###################################################################################################
#
# Utilities.py 		        (c) Benedikt Diemer
#							University of Chicago
#     				    	bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# Common routines for cosmocode modules.
#
###################################################################################################

import os
import configuration

###################################################################################################

def printLine():

	print('-------------------------------------------------------------------------------------')

	return

###################################################################################################

# Get a directory for the persistent caching of data. Here, this directory is chosen to be the
# directory where this file is located. This directory obviously already exists.

def getCacheDir():
	
	path = configuration.get_halotools_cache_dir() + '/'
	
	return path

###################################################################################################

# Returns the path to this code file.

def getCodeDir():

	return os.path.dirname(os.path.realpath(__file__))

###################################################################################################

# Tests whether the variable var is iterable or not.

def isArray(var):
	
	try:
		dummy = iter(var)
	except TypeError:
		ret = False
	else:
		ret = True
		
	return ret

###################################################################################################
