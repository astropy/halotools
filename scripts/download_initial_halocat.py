#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the default halo catalog"""

from halotools.sim_manager import CatalogManager, sim_defaults

simname = sim_defaults.default_simname
halo_finder = sim_defaults.default_halo_finder
redshift = sim_defaults.default_redshift

catman = CatalogManager()

catman.download_preprocessed_halo_catalog(simname = simname, 
	halo_finder = halo_finder, desired_redshift = redshift)


