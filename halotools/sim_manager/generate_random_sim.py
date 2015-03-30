# -*- coding: utf-8 -*-
"""

Simple module used to generate fake simulation data 
used to test the `~halotools.empirical_models` modules. 

"""
from astropy.table import Table
import numpy as np

__all__ = ['FakeSim']

class FakeSim(object):
	""" Fake simulation data used in the test suite of `~halotools.empirical_models`. 

	The `FakeSim` object has all the attributes required by 
	Mock Factories such as `~halotools.empirical_models.HodMockFactory` to 
	create a mock galaxy population. 
	The columns of the `halos` and `particles` attributes of `FakeSim` 
	are generated with ``np.random``. Thus mock catalogs built into `FakeSim` 
	will not have physically realistic spatial distributions, mass functions, etc.
	All the same, `FakeSim` is quite useful for testing purposes, 
	as it permits the testing of `~halotools.sim_manager` and `~halotools.empirical_models` 
	to be completely decoupled. Default behavior is to use a fixed seed in the random 
	number generation, so that an identical instance of `FakeSim` is created with each call. 
	"""

	def __init__(self, num_massbins = 6, num_halos_per_massbin = int(1e3), 
		num_ptcl = int(1e3), seed = 43):
		"""
		Parameters 
		----------
		num_massbins : int, optional 
			Number of distinct masses that will appear in the halo catalog. 
			Default is 6.

		num_halos_per_massbin : int, optional 
			Default is 1000

		num_ptcl : int, optional
			Number of dark matter particles. Default is 1000. 

		seed : int, optional 
			Random number seed used to generate the fake halos and particles. 
			Default is 43.
		"""
		self.Lbox = 250.0
		self.particle_mass = 1.e8
		self.simulation_name = 'fake'

		self.seed = seed

		self.num_massbins = num_massbins
		self.num_halos_per_massbin = num_halos_per_massbin
		self.num_halos = self.num_massbins*self.num_halos_per_massbin
		self.num_ptcl = num_ptcl

	@property 
	def halos(self):
		""" Astropy Table of randomly generated 
		dark matter halos. 
		"""

		np.random.seed(self.seed)

		haloid = np.arange(1e9, 1e9+self.num_halos)

		randomizer = np.random.random(self.num_halos)
		subhalo_fraction = 0.1
		upid = np.where(randomizer > subhalo_fraction, -1, 1)

		massbins = np.logspace(10, 15, self.num_massbins)
		mvir = np.repeat(massbins, self.num_halos_per_massbin, axis=0)
		logrvirbins = (np.log10(massbins) - 15)/3.
		rvir = np.repeat(10.**logrvirbins, self.num_halos_per_massbin, axis=0)
		logvmaxbins = -4.25 + 0.5*(np.log10(massbins) - logrvirbins)
		vmax = np.repeat(10.**logvmaxbins, self.num_halos_per_massbin, axis=0)

		conc = np.random.uniform(4, 15, self.num_halos)
		zhalf = np.random.uniform(0, 10, self.num_halos)

		pos = np.random.uniform(
			0, self.Lbox, self.num_halos*3).reshape(self.num_halos, 3)
		vel = np.random.uniform(
			-500, 500, self.num_halos*3).reshape(self.num_halos, 3)

		d = {
			'haloid': haloid, 
			'upid': upid, 
			'mvir': mvir, 
			'rvir': rvir, 
			'conc': conc, 
			'zhalf': zhalf, 
			'vmax': vmax, 
			'pos': pos, 
			'vel': vel
			}

		return Table(d)

	@property 
	def particles(self):
		""" Astropy Table of randomly generated 
		dark matter particles. 
		"""

		np.random.seed(self.seed)
		pos = np.random.uniform(
			0, self.Lbox, self.num_ptcl*3).reshape(self.num_ptcl, 3)
		d = {'pos': pos}

		return Table(d)

	




