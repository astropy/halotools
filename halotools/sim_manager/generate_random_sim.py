# -*- coding: utf-8 -*-
"""

Simple module used to generate fake simulation data 
used to test the `~halotools.empirical_models` modules. 

"""
from astropy.table import Table
import numpy as np

from .user_defined_halo_catalog import ScramScramScram

__all__ = ('FakeSim', )

class FakeSim(ScramScramScram):
	""" Fake simulation data used in the test suite of `~halotools.empirical_models`. 

	The `FakeSim` object has all the attributes required by 
	Mock Factories such as `~halotools.empirical_models.HodMockFactory` to 
	create a mock galaxy population. 
	The columns of the `halo_table` and `ptcl_table` attributes of `FakeSim` 
	are generated with ``np.random``. Thus mock catalogs built into `FakeSim` 
	will not have physically realistic spatial distributions, mass functions, etc.
	All the same, `FakeSim` is quite useful for testing purposes, 
	as it permits the testing of `~halotools.sim_manager` and `~halotools.empirical_models` 
	to be completely decoupled. Default behavior is to use a fixed seed in the random 
	number generation, so that an identical instance of `FakeSim` is created 
	for calls with the same arguments. 
	"""

	def __init__(self, num_massbins = 6, num_halos_per_massbin = int(1e3), 
		num_ptcl = int(1e4), seed = 43, redshift = 0.):
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
		Lbox = 250.0
		particle_mass = 1.e8
		simname = 'fake'

		self.seed = seed
		np.random.seed(self.seed)

		self.num_massbins = num_massbins
		self.num_halos_per_massbin = num_halos_per_massbin
		self.num_halos = self.num_massbins*self.num_halos_per_massbin
		self.num_ptcl = num_ptcl

		halo_id = np.arange(1e9, 1e9+self.num_halos)

		randomizer = np.random.random(self.num_halos)
		subhalo_fraction = 0.1
		upid = np.where(randomizer > subhalo_fraction, -1, 1)

		host_mask = upid == -1
		halo_hostid = np.zeros(len(halo_id))
		halo_hostid[host_mask] = halo_id[host_mask]
		halo_hostid[~host_mask] = upid[~host_mask]

		massbins = np.logspace(10, 15, self.num_massbins)
		mvir = np.repeat(massbins, self.num_halos_per_massbin, axis=0)
		mpeak = np.repeat(massbins, self.num_halos_per_massbin, axis=0)
		logrvirbins = (np.log10(massbins) - 15)/3.
		rvir = np.repeat(10.**logrvirbins, self.num_halos_per_massbin, axis=0)
		logvmaxbins = -4.25 + 0.5*(np.log10(massbins) - logrvirbins)
		vmax = np.repeat(10.**logvmaxbins, self.num_halos_per_massbin, axis=0)
		vpeak = vmax

		conc = np.random.uniform(4, 15, self.num_halos)
		rs = rvir/conc
		zhalf = np.random.uniform(0, 10, self.num_halos)

		x = np.random.uniform(0, Lbox, self.num_halos)
		y = np.random.uniform(0, Lbox, self.num_halos)
		z = np.random.uniform(0, Lbox, self.num_halos)
		vx = np.random.uniform(-500, 500, self.num_halos)
		vy = np.random.uniform(-500, 500, self.num_halos)
		vz = np.random.uniform(-500, 500, self.num_halos)

		px = np.random.uniform(0, Lbox, self.num_ptcl)
		py = np.random.uniform(0, Lbox, self.num_ptcl)
		pz = np.random.uniform(0, Lbox, self.num_ptcl)
		pvx = np.random.uniform(-1000, 1000, self.num_ptcl)
		pvy = np.random.uniform(-1000, 1000, self.num_ptcl)
		pvz = np.random.uniform(-1000, 1000, self.num_ptcl)
		d = {'x': px, 'y': py, 'z': pz, 'vx': pvx, 'vy': pvy, 'vz': pvz}
		ptcl_table = Table(d)


		ScramScramScram.__init__(self, 
			Lbox = Lbox, particle_mass = particle_mass, 
			redshift = 0.0, 
			halo_id = halo_id, 
			halo_x = x, halo_y = y, halo_z = z, 
			halo_vx = vx, halo_vy = vy, halo_vz = vz, 
			halo_upid = upid, 
			halo_hostid = halo_hostid, 
			halo_mvir = mvir, 
			halo_mpeak = mpeak, 
			halo_rvir = rvir, 
			halo_rs = rs, 
			halo_zhalf = zhalf, 
			halo_nfw_conc = conc, 
			halo_vmax = vmax, 
			halo_vpeak = vpeak, 
			ptcl_table = ptcl_table
			)

















