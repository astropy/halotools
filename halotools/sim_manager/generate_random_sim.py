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
		self.simname = 'fake'

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
		mpeak = np.repeat(massbins, self.num_halos_per_massbin, axis=0)
		logrvirbins = (np.log10(massbins) - 15)/3.
		rvir = np.repeat(10.**logrvirbins, self.num_halos_per_massbin, axis=0)
		logvmaxbins = -4.25 + 0.5*(np.log10(massbins) - logrvirbins)
		vmax = np.repeat(10.**logvmaxbins, self.num_halos_per_massbin, axis=0)

		conc = np.random.uniform(4, 15, self.num_halos)
		rs = rvir/conc
		zhalf = np.random.uniform(0, 10, self.num_halos)

		x = np.random.uniform(0, self.Lbox, self.num_halos)
		y = np.random.uniform(0, self.Lbox, self.num_halos)
		z = np.random.uniform(0, self.Lbox, self.num_halos)
		vx = np.random.uniform(-500, 500, self.num_halos)
		vy = np.random.uniform(-500, 500, self.num_halos)
		vz = np.random.uniform(-500, 500, self.num_halos)

		d = {
			'haloid': haloid, 
			'upid': upid, 
			'mvir': mvir, 
			'mpeak': mpeak, 
			'rvir': rvir, 
			'rs': rs, 
			'zhalf': zhalf, 
			'vmax': vmax, 
			'x': x, 
			'y': y, 
			'z': z, 
			'vx': vx, 
			'vy': vy, 
			'vz': vz
			}

		return Table(d)

	@property 
	def particles(self):
		""" Astropy Table of randomly generated 
		dark matter particles. 
		"""

		np.random.seed(self.seed)
		x = np.random.uniform(0, self.Lbox, self.num_ptcl)
		y = np.random.uniform(0, self.Lbox, self.num_ptcl)
		z = np.random.uniform(0, self.Lbox, self.num_ptcl)
		d = {'x': x, 'y': y, 'z':z}

		return Table(d)

	
class FakeMock(object):
	""" Fake galaxy data used in the test suite of `~halotools.empirical_models`. 
	"""

	def __init__(self, seed=43, approximate_ngals=2e4):
		""" 
		Parameters 
		----------
		approximate_ngals : int, optional 
			Approximate number of galaxies in the fake mock. Default is 2e4. 
		
		seed : int, optional 
			Random number seed used to generate the fake halos and particles. 
			Default is 43.

		Examples 
		--------
		Instantiating a randomly generated mock is simple: 

		>>> mock = FakeMock()

		The `FakeMock` object has all of the same basic structure as the 
		object produced by mock factories such as 
		`~halotools.empirical_models.HodMockFactory`. You can access the 
		collection of mock galaxies and their properties via ``mock.galaxy_table``:

		>>> ssfr_array = mock.galaxy_table['ssfr']

		Below are a couple of examples of how you might access various 
		subsets of the galaxy population. 

			* To select the population of *centrals*: 
				>>> central_mask = mock.galaxy_table['gal_type'] == 'centrals'
				>>> central_gals = mock.galaxy_table[central_mask]

				``central_gals`` is an Astropy `~astropy.table.Table` object, 
				so we can only access its attributes via column keys:

				>>> central_mstar = central_gals['stellar_mass']

			* To select the *cluster* population: 
				>>> cluster_mask = mock.galaxy_table['halo_mvir'] > 1e14
				>>> cluster_gals = mock.galaxy_table[cluster_mask]
				>>> cluster_ssfr = cluster_gals['ssfr']
		"""
		nhalos = np.max([100, approximate_ngals/20.]).astype(int)
		self.snapshot = FakeSim(num_halos_per_massbin=nhalos)
		self.halos = self.snapshot.halos
		self.particles = self.snapshot.particles
		self.create_astropy_table = True
		self.seed = seed

		self.gal_types = ['centrals', 'satellites', 'orphans']
		self._occupation_bounds = [1, float('inf'), float('inf')]

		self._set_fake_properties()
	
	def _set_fake_properties(self):

		np.random.seed(self.seed)

		central_occupations = np.random.random_integers(0,1,self.snapshot.num_halos)
		self.num_centrals = central_occupations.sum()
		central_nickname_array = np.repeat('centrals', self.num_centrals)
		central_halo_mvir = np.repeat(self.halos['mvir'], central_occupations)
		central_halo_haloid = np.repeat(self.halos['haloid'], central_occupations)
		central_halo_x = np.repeat(self.halos['x'], central_occupations, axis=0)
		central_halo_y = np.repeat(self.halos['y'], central_occupations, axis=0)
		central_halo_z = np.repeat(self.halos['z'], central_occupations, axis=0)
		central_halo_zhalf = np.repeat(self.halos['zhalf'], central_occupations)

		satellite_occupations = np.random.random_integers(0,3,self.snapshot.num_halos)
		self.num_satellites = satellite_occupations.sum()
		satellite_nickname_array = np.repeat('satellites', self.num_satellites)
		satellite_halo_mvir = np.repeat(self.halos['mvir'], satellite_occupations)
		satellite_halo_haloid = np.repeat(self.halos['haloid'], satellite_occupations)
		satellite_halo_x = np.repeat(self.halos['x'], satellite_occupations, axis=0)
		satellite_halo_y = np.repeat(self.halos['y'], satellite_occupations, axis=0)
		satellite_halo_z = np.repeat(self.halos['z'], satellite_occupations, axis=0)
		satellite_halo_zhalf = np.repeat(self.halos['zhalf'], satellite_occupations)

		censat_occ = np.append(central_occupations, satellite_occupations)
		censat_galtype = np.append(central_nickname_array, satellite_nickname_array)
		censat_halo_mvir = np.append(central_halo_mvir, satellite_halo_mvir)
		censat_halo_haloid = np.append(central_halo_haloid, satellite_halo_haloid)
		censat_halo_x = np.append(central_halo_x, satellite_halo_x, axis=0)
		censat_halo_y = np.append(central_halo_y, satellite_halo_y, axis=0)
		censat_halo_z = np.append(central_halo_z, satellite_halo_z, axis=0)
		censat_halo_zhalf = np.append(central_halo_zhalf, satellite_halo_zhalf)

		orphan_occupations = np.random.random_integers(0,3,self.snapshot.num_halos)
		self.num_orphans = orphan_occupations.sum()
		orphan_nickname_array = np.repeat('orphans', self.num_orphans)
		orphan_halo_mvir = np.repeat(self.halos['mvir'], orphan_occupations)
		orphan_halo_haloid = np.repeat(self.halos['haloid'], orphan_occupations)
		orphan_halo_x = np.repeat(self.halos['x'], orphan_occupations, axis=0)
		orphan_halo_y = np.repeat(self.halos['y'], orphan_occupations, axis=0)
		orphan_halo_z = np.repeat(self.halos['z'], orphan_occupations, axis=0)
		orphan_halo_zhalf = np.repeat(self.halos['zhalf'], orphan_occupations)

		self.galaxy_table = Table()

		#self._occupation = np.append(censat_occ, orphan_occupations)
		self.galaxy_table['gal_type'] = np.append(
			censat_galtype, orphan_nickname_array)
		self.galaxy_table['halo_mvir'] = np.append(
			censat_halo_mvir, orphan_halo_mvir)
		self.galaxy_table['halo_haloid'] = np.append(
			censat_halo_haloid, orphan_halo_haloid).astype(int)
		self.galaxy_table['halo_x'] = np.append(censat_halo_x, orphan_halo_x, axis=0)
		self.galaxy_table['halo_y'] = np.append(censat_halo_y, orphan_halo_y, axis=0)
		self.galaxy_table['halo_z'] = np.append(censat_halo_z, orphan_halo_z, axis=0)
		self.galaxy_table['halo_zhalf'] = np.append(censat_halo_zhalf, orphan_halo_zhalf)

		num_gals = self.num_centrals + self.num_satellites + self.num_orphans 

		self.galaxy_table['x'] = np.random.uniform(0, self.snapshot.Lbox, num_gals)
		self.galaxy_table['y'] = np.random.uniform(0, self.snapshot.Lbox, num_gals)
		self.galaxy_table['z'] = np.random.uniform(0, self.snapshot.Lbox, num_gals)

		self.galaxy_table['stellar_mass'] = 10.**(np.random.power(2, size=num_gals)*3 + 9)

		def get_colors(num_gals, red_fraction):
		    num_red = np.round(num_gals*red_fraction).astype(int)
		    num_blue = num_gals - num_red
		    red_sequence = np.random.normal(loc=0.75, scale=0.1, size=num_red)
		    blue_cloud = np.random.normal(loc=0, scale=0.2, size=num_blue)
		    colors = np.append(red_sequence, blue_cloud)
		    return colors

		def get_ssfr(num_gals, red_fraction):
		    num_red = np.round(num_gals*red_fraction).astype(int)
		    num_blue = num_gals - num_red
		    red_sequence = np.random.normal(loc=-11.5, scale=0.3, size=num_red)
		    blue_cloud = np.random.normal(loc=-9.5, scale=0.4, size=num_blue)
		    ssfr = np.append(red_sequence, blue_cloud)
		    return ssfr

		def get_bulge_to_disk_ratio(num_gals, bulge_fraction):
			bulge_mass = np.random.normal(loc=1, scale=0.5, size=num_gals)
			disk_fraction = 1-bulge_fraction
			num_disk = np.round(num_gals*disk_fraction).astype(int)
			num_bulge = num_gals - num_disk
			bulge = np.random.uniform(0.25, 1.5, num_bulge)
			disk = np.random.uniform(0.75, 5, num_disk)
			disk_mass = np.append(bulge, disk)
			ratio = bulge_mass/disk_mass
			ratio = np.where(ratio < 0.01, 1, ratio)
			return ratio

		sm_min = self.galaxy_table['stellar_mass'].min()
		sm_max = self.galaxy_table['stellar_mass'].max()
		sm_bins = np.logspace(np.log10(sm_min)-0.01, np.log10(sm_max)+0.01, 50)
		bimodality = np.linspace(0.1, 0.9, len(sm_bins))

		colors = np.zeros(num_gals)
		ssfr = np.zeros(num_gals)
		bulge_ratios = np.zeros(num_gals)

		for ibin in range(len(sm_bins)-1):
			idx_bini = np.where(
				(self.galaxy_table['stellar_mass'] > sm_bins[ibin]) & 
				(self.galaxy_table['stellar_mass'] < sm_bins[ibin+1]))[0]
			num_gals_ibin = len(self.galaxy_table[idx_bini])

			if num_gals_ibin > 0:
				colors_bini = get_colors(
					num_gals_ibin, bimodality[ibin])
				ssfr_bini = get_ssfr(
					num_gals_ibin, bimodality[ibin])
				bulge_ratios_bini = get_bulge_to_disk_ratio(
					num_gals_ibin, bimodality[ibin])

				colors[idx_bini] = colors_bini
				ssfr[idx_bini] = ssfr_bini
				bulge_ratios[idx_bini] = bulge_ratios_bini


		self.galaxy_table['bulge_to_disk_ratio'] = bulge_ratios
		self.galaxy_table['ssfr'] = ssfr
		self.galaxy_table['gr_color'] = colors



















