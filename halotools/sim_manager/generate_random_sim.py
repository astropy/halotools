# -*- coding: utf-8 -*-
"""

Simple module used to generate fake simulation data 
used to test the `~halotools.empirical_models` modules. 

"""
from astropy.table import Table
import numpy as np

__all__ = ['FakeSim', 'FakeMock']

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

	
class FakeMock(object):
	""" Fake galaxy data used in the test suite of `~halotools.empirical_models`. 
	"""

	def __init__(self, seed=43):
		""" 
		Parameters 
		----------
		seed : int, optional 
			Random number seed used to generate the fake halos and particles. 
			Default is 43.

		Examples 
		--------
		>>> mock = FakeMock()
		>>> central_gals = mock.galaxy_table[mock.gal_type == 'centrals']
		>>> quenched_gals = mock.galaxy_table[mock.ssfr < -11]
		>>> quenched_orphan_mask = (mock.gal_type == 'orphans') & (mock.ssfr < -11)
		>>> quenched_orphan_gals = mock.galaxy_table[quenched_orphan_mask]

		"""
		self.snapshot = FakeSim()
		self.halos = self.snapshot.halos
		self.particles = self.snapshot.particles
		self.create_astropy_table = True
		self.seed = seed

		self.gal_types = ['centrals', 'satellites', 'orphans']
		self._occupation_bounds = [1, float('inf'), float('inf')]

		self._set_fake_properties()

	@property 
	def galaxy_table(self):
		""" Astropy Table storing mock galaxy data. 
		"""

		d = {'gal_type' : self.gal_type, 
			'halo_mvir' : self.halo_mvir, 
			'halo_haloid' : self.halo_haloid, 
			'halo_pos' : self.halo_pos,
			'halo_zhalf' : self.halo_zhalf, 
			'mstar' : self.mstar, 
			'ssfr' : self.ssfr
		}

		return Table(d)

	
	def _set_fake_properties(self):

		np.random.seed(self.seed)

		central_occupations = np.random.random_integers(0,1,self.snapshot.num_halos)
		self.num_centrals = central_occupations.sum()
		central_nickname_array = np.repeat('centrals', self.num_centrals)
		central_halo_mvir = np.repeat(self.halos['mvir'], central_occupations)
		central_halo_haloid = np.repeat(self.halos['haloid'], central_occupations)
		central_halo_pos = np.repeat(self.halos['pos'], central_occupations, axis=0)
		central_halo_zhalf = np.repeat(self.halos['zhalf'], central_occupations)

		satellite_occupations = np.random.random_integers(0,3,self.snapshot.num_halos)
		self.num_satellites = satellite_occupations.sum()
		satellite_nickname_array = np.repeat('satellites', self.num_satellites)
		satellite_halo_mvir = np.repeat(self.halos['mvir'], satellite_occupations)
		satellite_halo_haloid = np.repeat(self.halos['haloid'], satellite_occupations)
		satellite_halo_pos = np.repeat(self.halos['pos'], satellite_occupations, axis=0)
		satellite_halo_zhalf = np.repeat(self.halos['zhalf'], satellite_occupations)

		censat_occ = np.append(central_occupations, satellite_occupations)
		censat_galtype = np.append(central_nickname_array, satellite_nickname_array)
		censat_halo_mvir = np.append(central_halo_mvir, satellite_halo_mvir)
		censat_halo_haloid = np.append(central_halo_haloid, satellite_halo_haloid)
		censat_halo_pos = np.append(central_halo_pos, satellite_halo_pos, axis=0)
		censat_halo_zhalf = np.append(central_halo_zhalf, satellite_halo_zhalf)

		orphan_occupations = np.random.random_integers(0,3,self.snapshot.num_halos)
		self.num_orphans = orphan_occupations.sum()
		orphan_nickname_array = np.repeat('orphans', self.num_orphans)
		orphan_halo_mvir = np.repeat(self.halos['mvir'], orphan_occupations)
		orphan_halo_haloid = np.repeat(self.halos['haloid'], orphan_occupations)
		orphan_halo_pos = np.repeat(self.halos['pos'], orphan_occupations, axis=0)
		orphan_halo_zhalf = np.repeat(self.halos['zhalf'], orphan_occupations)

		self._occupation = np.append(censat_occ, orphan_occupations)
		self.gal_type = np.append(censat_galtype, orphan_nickname_array)
		self.halo_mvir = np.append(censat_halo_mvir, orphan_halo_mvir)
		self.halo_haloid = np.append(censat_halo_haloid, orphan_halo_haloid).astype(int)
		self.halo_pos = np.append(censat_halo_pos, orphan_halo_pos, axis=0)
		self.halo_zhalf = np.append(censat_halo_zhalf, orphan_halo_zhalf)

		self.num_gals = self.num_centrals + self.num_satellites + self.num_orphans 
		self.mstar = np.random.uniform(8, 12, self.num_gals)
		self.ssfr = np.random.uniform(-12, -9, self.num_gals)


















