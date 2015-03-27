# -*- coding: utf-8 -*-
"""

Simple module used to generate fake simulation data 
used to test the `~halotools.empirical_models` modules. 

"""
from astropy.table import Table
import numpy as np

__all__ = ['FakeSim']

class FakeSim(object):

	def __init__(self, 
		num_massbins = 6, num_halos_per_massbin = int(1e3), num_ptcl = int(1e3), 
		seed = 43):

		self.Lbox = 250.0
		self.particle_mass = 1.e8
		self.simulation_name = 'fake'

		self.seed = 43

		self.num_massbins = num_massbins
		self.num_halos_per_massbin = num_halos_per_massbin
		self.num_halos = self.num_massbins*self.num_halos_per_massbin
		self.num_ptcl = num_ptcl



	@property 
	def halos(self):

		haloid = np.arange(1e9, 1e9+self.num_halos)

		np.random.seed(self.seed)
		randomizer = np.random.random(self.num_halos)
		subhalo_fraction = 0.1
		upid = np.where(randomizer > subhalo_fraction, -1, 1)

		massbins = np.logspace(10, 15, self.num_massbins)
		mvir = np.repeat(massbins, self.num_halos_per_massbin, axis=0)
		logrvirbins = (np.log10(massbins) - 15)/3.
		rvir = np.repeat(10.**logrvirbins, self.num_halos_per_massbin, axis=0)

		conc = np.random.uniform(4, 15, self.num_halos)
		last_mm = np.random.uniform(0, 1, self.num_halos)

		logvmaxbins = -4.25 + 0.5*(np.log10(massbins) - logrvirbins)
		vmax = np.repeat(10.**logvmaxbins, self.num_halos_per_massbin, axis=0)

		np.random.seed(self.seed)
		pos = np.random.uniform(
			0, self.Lbox, self.num_halos*3).reshape(self.num_halos, 3)
		np.random.seed(self.seed)
		vel = np.random.uniform(
			-500, 500, self.num_halos*3).reshape(self.num_halos, 3)

		d = {
			'haloid': haloid, 
			'upid': upid, 
			'mvir': mvir, 
			'rvir': rvir, 
			'conc': conc, 
			'last_mm': last_mm, 
			'vmax': vmax, 
			'pos': pos, 
			'vel': vel
			}

		return Table(d)


	




