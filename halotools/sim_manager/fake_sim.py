"""
Module storing the `~halotools.sim_manager.FakeSim` class
used to generate fake simulation data on-the-fly.
Primary use is to test the `~halotools.empirical_models` modules,
particularly with doctests.
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .user_supplied_halo_catalog import UserSuppliedHaloCatalog
from .user_supplied_ptcl_catalog import UserSuppliedPtclCatalog
from .sim_defaults import default_cosmology

from ..utils import crossmatch

__all__ = ('FakeSim', 'FakeSimHalosNearBoundaries')


class FakeSim(UserSuppliedHaloCatalog):
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

    def __init__(self, num_massbins=10, num_halos_per_massbin=int(100),
            num_ptcl=int(2e4), seed=43, redshift=0., cosmology=default_cosmology,
            **kwargs):
        """
        Parameters
        ----------
        num_massbins : int, optional
            Number of distinct masses that will appear in the halo catalog.
            Default is 10.

        num_halos_per_massbin : int, optional
            Default is 100

        num_ptcl : int, optional
            Number of dark matter particles. Default is 20000.

        seed : int, optional
            Random number seed used to generate the fake halos and particles.
            Default is 43.

        cosmology : `astropy.cosmology` instance, optional
            Default is `halotools.sim_manager.sim_defaults.default_cosmology`.

        Examples
        --------
        >>> halocat = FakeSim()

        """
        Lbox = 250.0
        particle_mass = 1.e8
        try:
            self.simname = kwargs['simname']
        except KeyError:
            self.simname = 'fake'
        try:
            self.halo_finder = kwargs['halo_finder']
        except KeyError:
            self.halo_finder = 'fake'
        try:
            self.version_name = kwargs['version_name']
        except KeyError:
            self.version_name = 'dummy_version'
        self.redshift = redshift

        self.seed = seed

        self.num_massbins = num_massbins
        self.num_halos_per_massbin = num_halos_per_massbin
        self.num_halos = self.num_massbins*self.num_halos_per_massbin

        approx_num_ptcl = num_ptcl
        self.num_ptcl_per_dim = int(approx_num_ptcl**(1/3.))
        self.num_ptcl = self.num_ptcl_per_dim**3

        self.cosmology = cosmology

        halo_id = np.arange(1e5, 1e5+2*self.num_halos, dtype='i8')
        with NumpyRNGContext(self.seed):
            np.random.shuffle(halo_id)
        halo_id = halo_id[:self.num_halos]

        with NumpyRNGContext(self.seed):
            randomizer = np.random.random(self.num_halos)
        subhalo_fraction = 0.1
        upid = np.where(randomizer > subhalo_fraction, -1, 1)

        host_mask = upid == -1
        host_ids = halo_id[host_mask]
        with NumpyRNGContext(self.seed):
            upid[~host_mask] = np.random.choice(host_ids, len(upid[~host_mask]))

        halo_hostid = np.zeros(len(halo_id), dtype='i8')
        halo_hostid[host_mask] = halo_id[host_mask]
        halo_hostid[~host_mask] = upid[~host_mask]

        massbins = np.logspace(10, 16, self.num_massbins)
        mvir = np.repeat(massbins, self.num_halos_per_massbin, axis=0)
        mpeak = np.repeat(massbins, self.num_halos_per_massbin, axis=0)
        logrvirbins = (np.log10(massbins) - 15)/3.
        rvir = np.repeat(10.**logrvirbins, self.num_halos_per_massbin, axis=0)
        logvmaxbins = -4.25 + 0.5*(np.log10(massbins) - logrvirbins)
        vmax = np.repeat(10.**logvmaxbins, self.num_halos_per_massbin, axis=0)
        vpeak = vmax

        with NumpyRNGContext(self.seed):
            spin = np.random.random(self.num_halos)
            conc = np.random.uniform(4, 15, self.num_halos)
            rs = rvir/conc
            zhalf = np.random.uniform(0, 10, self.num_halos)
            dmdt = np.random.uniform(-1, 10, self.num_halos)

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

        idxA, idxB = crossmatch(halo_hostid, halo_id)
        halo_mvir_host_halo = np.copy(mvir)
        halo_mvir_host_halo[idxA] = mvir[idxB]

        ptclcat = UserSuppliedPtclCatalog(
            Lbox=Lbox, redshift=redshift, particle_mass=particle_mass,
            x=px, y=py, z=pz, vx=pvx, vy=pvy, vz=pvz)

        UserSuppliedHaloCatalog.__init__(self,
            Lbox=Lbox, particle_mass=particle_mass,
            redshift=redshift,
            halo_id=halo_id,
            halo_x=x, halo_y=y, halo_z=z,
            halo_vx=vx, halo_vy=vy, halo_vz=vz,
            halo_upid=upid,
            halo_hostid=halo_hostid,
            halo_mvir=mvir,
            halo_mpeak=mpeak,
            halo_m200b=mvir,
	    halo_m180b=mvir,
            halo_rvir=rvir,
            halo_rs=rs,
            halo_zhalf=zhalf,
            halo_nfw_conc=conc,
            halo_vmax=vmax,
            halo_vpeak=vpeak,
            halo_spin=spin,
            halo_mass_accretion_rate=dmdt,
            halo_mvir_host_halo=halo_mvir_host_halo,
            user_supplied_ptclcat=ptclcat
                                         )


class FakeSimHalosNearBoundaries(UserSuppliedHaloCatalog):
    """ Fake simulation data used in the test suite of `~halotools.empirical_models`.

    The only difference between `FakeSim` and `FakeSimHalosNearBoundaries`
    is that all halos reside right near the very edge of the box.
    Useful for unit-testing the treatment of periodic boundary conditions.
    """

    def __init__(self, num_massbins=6, num_halos_per_massbin=int(100),
            num_ptcl=int(1e4), seed=43, redshift=0., **kwargs):
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

        Examples
        --------
        >>> fakesim = FakeSimHalosNearBoundaries()
        """
        Lbox = 250.0
        particle_mass = 1.e8
        self.simname = 'fake'
        self.halo_finder = 'fake'
        self.version_name = 'dummy_version'

        self.seed = seed

        self.num_massbins = num_massbins
        self.num_halos_per_massbin = num_halos_per_massbin
        self.num_halos = self.num_massbins*self.num_halos_per_massbin
        self.num_ptcl = num_ptcl

        halo_id = np.arange(1e9, 1e9+self.num_halos)

        with NumpyRNGContext(self.seed):
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

        with NumpyRNGContext(self.seed):
            conc = np.random.uniform(4, 15, self.num_halos)
            rs = rvir/conc
            zhalf = np.random.uniform(0, 10, self.num_halos)

        x = np.zeros(self.num_halos)
        y = np.zeros(self.num_halos)
        z = np.zeros(self.num_halos)
        middle_index = int(self.num_halos/2.)

        with NumpyRNGContext(self.seed):
            x[:middle_index] = np.random.uniform(0, 0.001, len(x[:middle_index]))
            x[middle_index:] = np.random.uniform(Lbox-0.001, Lbox, len(x[middle_index:]))
            y[:middle_index] = np.random.uniform(0, 0.001, len(y[:middle_index]))
            y[middle_index:] = np.random.uniform(Lbox-0.001, Lbox, len(y[middle_index:]))
            z[:middle_index] = np.random.uniform(0, 0.001, len(z[:middle_index]))
            z[middle_index:] = np.random.uniform(Lbox-0.001, Lbox, len(z[middle_index:]))

            vx = np.random.uniform(-500, 500, self.num_halos)
            vy = np.random.uniform(-500, 500, self.num_halos)
            vz = np.random.uniform(-500, 500, self.num_halos)

            px = np.random.uniform(0, Lbox, self.num_ptcl)
            py = np.random.uniform(0, Lbox, self.num_ptcl)
            pz = np.random.uniform(0, Lbox, self.num_ptcl)
            pvx = np.random.uniform(-1000, 1000, self.num_ptcl)
            pvy = np.random.uniform(-1000, 1000, self.num_ptcl)
            pvz = np.random.uniform(-1000, 1000, self.num_ptcl)

        ptclcat = UserSuppliedPtclCatalog(
            Lbox=Lbox, redshift=redshift, particle_mass=particle_mass,
            x=px, y=py, z=pz, vx=pvx, vy=pvy, vz=pvz)

        UserSuppliedHaloCatalog.__init__(self,
            Lbox=Lbox, particle_mass=particle_mass,
            redshift=redshift,
            halo_id=halo_id,
            halo_x=x, halo_y=y, halo_z=z,
            halo_vx=vx, halo_vy=vy, halo_vz=vz,
            halo_upid=upid,
            halo_hostid=halo_hostid,
            halo_mvir=mvir,
            halo_mpeak=mpeak,
            halo_rvir=rvir,
            halo_rs=rs,
            halo_zhalf=zhalf,
            halo_nfw_conc=conc,
            halo_vmax=vmax,
            halo_vpeak=vpeak,
            user_supplied_ptclcat=ptclcat
                                         )
