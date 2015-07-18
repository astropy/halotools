#!/usr/bin/env python

import os
from ..catalog_manager import CatalogManager

def test_processed_halocats_in_cache():
	temp_fnames = []
	temp_fnames.append(os.path.abspath('dummy_sim1/dummy_hfinder1/dummy_fname1.dummyversion1.hdf5'))
	temp_fnames.append(os.path.abspath('dummy_sim1/dummy_hfinder1/dummy_fname2.dummyversion1.hdf5'))
	temp_fnames.append(os.path.abspath('dummy_sim1/dummy_hfinder1/dummy_fname3.dummyversion1.hdf5'))

	temp_fnames.append(os.path.abspath('dummy_sim2/dummy_hfinder1/dummy_fname1.dummyversion1.hdf5'))
	temp_fnames.append(os.path.abspath('dummy_sim2/dummy_hfinder1/dummy_fname1.dummyversion2.hdf5'))
	temp_fnames.append(os.path.abspath('dummy_sim2/dummy_hfinder1/dummy_fname2.dummyversion1.hdf5'))
	temp_fnames.append(os.path.abspath('dummy_sim2/dummy_hfinder2/dummy_fname1.dummyversion1.hdf5'))

	for fname in temp_fnames:
		os.system('touch ' + fname)

	catman = CatalogManager()
	






