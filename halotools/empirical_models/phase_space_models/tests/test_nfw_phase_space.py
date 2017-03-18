"""
"""
from unittest import TestCase
import numpy as np
from copy import deepcopy
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext

from ..nfw_phase_space import NFWPhaseSpace
from ..profile_models.tests import analytic_nfw_density_outer_shell_normalization
from ..profile_models.tests import monte_carlo_density_outer_shell_normalization

__all__ = ['TestNFWPhaseSpace']

fixed_seed = 43


class TestNFWPhaseSpace(TestCase):
    """ Class used to test `~halotools.empirical_models.NFWPhaseSpace`.
    """

    def setUp(self):
        """ Load the NFW model and build a coarse lookup table.
        """
        self.nfw = NFWPhaseSpace()
        cmin, cmax, dc = 1, 25, 0.5
        self.nfw.setup_prof_lookup_tables((cmin, cmax, dc))
        self.nfw.build_lookup_tables()

        Npts = int(5e4)
        self.c15 = np.zeros(Npts) + 15
        self.c10 = np.zeros(Npts) + 10
        self.c5 = np.zeros(Npts) + 5

        npts = int(1e3)
        Lbox = 250
        zeros = np.zeros(npts)

        with NumpyRNGContext(fixed_seed):
            x = np.random.uniform(0, Lbox, npts)
        with NumpyRNGContext(fixed_seed+1):
            y = np.random.uniform(0, Lbox, npts)
        with NumpyRNGContext(fixed_seed+2):
            z = np.random.uniform(0, Lbox, npts)
        with NumpyRNGContext(fixed_seed+3):
            halo_vx = np.random.uniform(-250, 250, npts)
        with NumpyRNGContext(fixed_seed+4):
            halo_vy = np.random.uniform(-250, 250, npts)
        with NumpyRNGContext(fixed_seed+5):
            halo_vz = np.random.uniform(-250, 250, npts)
        with NumpyRNGContext(fixed_seed+6):
            d = np.random.uniform(0, 0.25, npts)
            rvir = np.zeros(npts) + 0.2
        with NumpyRNGContext(fixed_seed+7):
            conc_nfw = np.random.uniform(1.5, 15, npts)
        mass = np.zeros(npts) + 1e12

        self._dummy_halo_table = Table({'halo_x': x, 'halo_y': y, 'halo_z': z,
            'host_centric_distance': d, 'halo_rvir': rvir, 'conc_NFWmodel': conc_nfw,
            'halo_vx': halo_vx, 'halo_vy': halo_vy, 'halo_vz': halo_vz, 'halo_mvir': mass,
            'x': zeros, 'y': zeros, 'z': zeros, 'vx': halo_vx, 'vy': halo_vy, 'vz': halo_vz})

    def test_constructor(self):
        """ Test that composite phase space models have all the appropriate attributes.
        """
        # MonteCarloGalProf attributes
        assert hasattr(self.nfw, 'logradius_array')
        assert hasattr(self.nfw, 'rad_prof_func_table')
        assert hasattr(self.nfw, 'vel_prof_func_table')
        assert hasattr(self.nfw, '_mc_dimensionless_radial_distance')

        # NFWPhaseSpace attributes
        assert hasattr(self.nfw, 'assign_phase_space')
        assert hasattr(self.nfw, '_galprop_dtypes_to_allocate')

        # AnalyticDensityProf attributes
        assert hasattr(self.nfw, 'circular_velocity')

        # NFWProfile attributes
        assert hasattr(self.nfw, 'mass_density')
        assert hasattr(self.nfw, 'prof_param_keys')

        # ConcMass
        assert hasattr(self.nfw, 'conc_NFWmodel')
        assert hasattr(self.nfw, 'conc_mass_model')

    def test_mc_unit_sphere(self):
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_unit_sphere`.

        This test verifies that all returned 3d points are at unit distance from the origin.
        """
        x, y, z = self.nfw.mc_unit_sphere(100, seed=43)
        pos = np.vstack([x, y, z]).T
        norm = np.linalg.norm(pos, axis=1)
        assert np.allclose(norm, 1, rtol=1e-4)

    def test_mc_unit_sphere_stochasticity(self):
        """ Method used to test correctness of stochasticity/deterministic behavior of
        `~halotools.empirical_models.NFWPhaseSpace.mc_unit_sphere`.
        """
        x1, y1, z1 = self.nfw.mc_unit_sphere(100, seed=43)
        x2, y2, z2 = self.nfw.mc_unit_sphere(100, seed=43)
        x3, y3, z3 = self.nfw.mc_unit_sphere(100, seed=None)
        assert np.allclose(x1, x2, rtol=0.001)
        assert not np.allclose(x1, x3, rtol=0.001)

    def test_mc_dimensionless_radial_distance(self):
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`.

        Method uses the `~halotools.empirical_models.analytic_nfw_density_outer_shell_normalization` function
        and the `~halotools.empirical_models.monte_carlo_density_outer_shell_normalization` function
        to verify that the points returned by `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`
        do indeed trace an NFW profile.

        """

        r15 = self.nfw._mc_dimensionless_radial_distance(self.c15, seed=43)
        r10 = self.nfw._mc_dimensionless_radial_distance(self.c10, seed=43)
        r5 = self.nfw._mc_dimensionless_radial_distance(self.c5, seed=43)

        assert np.all(r15 <= 1)
        assert np.all(r15 >= 0)
        assert np.all(r10 <= 1)
        assert np.all(r10 >= 0)
        assert np.all(r5 <= 1)
        assert np.all(r5 >= 0)

        assert np.mean(r15) < np.mean(r10) < np.mean(r5)
        assert np.median(r15) < np.median(r10) < np.median(r5)

        num_rbins = 15
        rbins = np.linspace(0.05, 1, num_rbins)
        for r, c in zip([r5, r10, r15], [5, 10, 15]):
            rbin_midpoints, monte_carlo_ratio = (
                monte_carlo_density_outer_shell_normalization(rbins, r))
            analytical_ratio = (
                analytic_nfw_density_outer_shell_normalization(rbin_midpoints, c))
            assert np.allclose(monte_carlo_ratio, analytical_ratio, 0.05)

    def test_mc_dimensionless_radial_distance_stochasticity(self):
        """ Method used to test correctness of stochasticity/deterministic behavior of
        `~halotools.empirical_models.NFWPhaseSpace._mc_dimensionless_radial_distance`.
        """
        x1 = self.nfw._mc_dimensionless_radial_distance(self.c15, seed=43)
        x2 = self.nfw._mc_dimensionless_radial_distance(self.c15, seed=43)
        x3 = self.nfw._mc_dimensionless_radial_distance(self.c15, seed=None)
        assert np.allclose(x1, x2, rtol=0.001)
        assert not np.allclose(x1, x3, rtol=0.001)

    def test_mc_solid_sphere(self):
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_solid_sphere`.

        Method ensures that all returned points lie inside the unit sphere.
        """
        x, y, z = self.nfw.mc_solid_sphere(self.c15, seed=43)
        pos = np.vstack([x, y, z]).T
        norm = np.linalg.norm(pos, axis=1)
        assert np.all(norm < 1)
        assert np.all(norm > 0)
        assert np.all(x > -1)
        assert np.all(x < 1)
        assert np.all(y > -1)
        assert np.all(y < 1)
        assert np.all(z > -1)
        assert np.all(z < 1)

    def test_mc_solid_sphere_stochasticity(self):
        """ Method used to test correctness of stochasticity/deterministic behavior of
        `~halotools.empirical_models.NFWPhaseSpace.mc_solid_sphere`.
        """
        x1, y1, z1 = self.nfw.mc_solid_sphere(self.c15, seed=43)
        x2, y2, z2 = self.nfw.mc_solid_sphere(self.c15, seed=43)
        x3, y3, z3 = self.nfw.mc_solid_sphere(self.c15, seed=None)

        assert np.allclose(x1, x2, rtol=0.001)
        assert not np.allclose(x1, x3, rtol=0.001)

    def test_mc_halo_centric_pos(self):
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`.

        Method verifies

        1. All returned points lie within the correct radial distance

        2. Increasing the input concentration decreases the mean and median radial distance of the returned points.

        """
        r = 0.25
        halo_radius = np.zeros(len(self.c15)) + r
        x15, y15, z15 = self.nfw.mc_halo_centric_pos(self.c15,
            halo_radius=halo_radius, seed=43)
        assert np.all(x15 > -r)
        assert np.all(x15 < r)
        assert np.all(y15 > -r)
        assert np.all(y15 < r)
        assert np.all(z15 > -r)
        assert np.all(z15 < r)

        pos15 = np.vstack([x15, y15, z15]).T
        norm15 = np.linalg.norm(pos15, axis=1)
        assert np.all(norm15 < r)
        assert np.all(norm15 > 0)

        x5, y5, z5 = self.nfw.mc_halo_centric_pos(self.c5,
            halo_radius=halo_radius, seed=43)
        pos5 = np.vstack([x5, y5, z5]).T
        norm5 = np.linalg.norm(pos5, axis=1)

        x10, y10, z10 = self.nfw.mc_halo_centric_pos(self.c10,
            halo_radius=halo_radius,  seed=43)
        pos10 = np.vstack([x10, y10, z10]).T
        norm10 = np.linalg.norm(pos10, axis=1)

        assert np.mean(norm5) > np.mean(norm10)
        assert np.mean(norm10) > np.mean(norm15)

        assert np.median(norm5) > np.median(norm10)
        assert np.median(norm10) > np.median(norm15)

        x10a, y10a, z10a = self.nfw.mc_halo_centric_pos(self.c10,
            halo_radius=halo_radius*2, seed=43)
        pos10a = np.vstack([x10a, y10a, z10a]).T
        norm10a = np.linalg.norm(pos10a, axis=1)

        assert np.any(norm10a > r)
        assert np.all(norm10a < 2*r)

        t = Table({'c': self.c15})

    def test_mc_halo_centric_pos_stochasticity(self):
        """ Method used to test stochasticity/deterministic behavior of
        `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`.

        """
        r = 0.25
        halo_radius = np.zeros(len(self.c15)) + r
        x15a, y15a, z15a = self.nfw.mc_halo_centric_pos(self.c15,
            halo_radius=halo_radius, seed=43)
        x15b, y15b, z15b = self.nfw.mc_halo_centric_pos(self.c15,
            halo_radius=halo_radius, seed=43)
        x15c, y15c, z15c = self.nfw.mc_halo_centric_pos(self.c15,
            halo_radius=halo_radius, seed=None)
        assert np.allclose(x15a, x15b, rtol=0.01)
        assert not np.allclose(x15a, x15c, rtol=0.01)

    def test_mc_pos(self):
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_halo_centric_pos`.

        Method verifies that passing an input ``seed`` results in deterministic behavior.

        Notes
        -----
        Clearly this particular function would benefit from more robust unit-testing.

        """
        r = 0.25
        halo_radius = np.zeros(len(self.c15)) + r
        x1, y1, z1 = self.nfw.mc_pos(self.c15,
            halo_radius=halo_radius, seed=43)
        x2, y2, z2 = self.nfw.mc_halo_centric_pos(self.c15,
            halo_radius=halo_radius, seed=43)
        assert np.all(x1 == x2)
        assert np.all(y1 == y2)
        assert np.all(z1 == z2)

        self.nfw.mc_pos(table=self._dummy_halo_table)

    def test_vrad_disp_from_lookup(self):
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace._vrad_disp_from_lookup`.

        Method verifies that all scaled velocities are between zero and unity.

        Notes
        -----
        Clearly this particular function would benefit from more robust unit-testing.

        """
        with NumpyRNGContext(fixed_seed):
            scaled_radius = np.random.uniform(0, 1, len(self.c15))
        vr_disp = self.nfw._vrad_disp_from_lookup(scaled_radius, self.c15, seed=43)

        assert np.all(vr_disp < 1)
        assert np.all(vr_disp > 0)

    def test_mc_radial_velocity(self):
        """ Method used to test `~halotools.empirical_models.NFWPhaseSpace.mc_radial_velocity`.

        Method generates a Monte Carlo velocity profile realization with all points at :math:`R_{\\rm max}`,
        and compares the manually computed velocity dispersion to the analytical expectation for :math:`V_{\\rm max}`,
        as computed by the `~halotools.empirical_models.NFWPhaseSpace.vmax`. Method
        verifies that these two results agree within the expected random noise level.
        """
        npts = int(1e4)
        conc = 10
        carr = np.zeros(npts) + conc

        mass = 1e12
        rmax = self.nfw.rmax(mass, conc)
        vmax = self.nfw.vmax(mass, conc)
        r = np.zeros(npts) + rmax
        rvir = self.nfw.halo_mass_to_halo_radius(mass)
        scaled_radius = r/rvir

        mc_vr = self.nfw.mc_radial_velocity(scaled_radius, mass, carr, seed=43)
        vr_dispersion_from_monte_carlo = np.std(mc_vr)

        analytical_result = vmax[0]/np.sqrt(3.)
        assert np.allclose(vr_dispersion_from_monte_carlo, analytical_result, rtol=0.1)

    def test_mc_radial_velocity_stochasticity(self):
        """ Method used to verify correct deterministic/stochastic behavior of
        `~halotools.empirical_models.NFWPhaseSpace.mc_radial_velocity`.

        """
        npts = int(1e4)
        conc = 10
        carr = np.zeros(npts) + conc

        mass = 1e12
        rmax = self.nfw.rmax(mass, conc)
        r = np.zeros(npts) + rmax
        rvir = self.nfw.halo_mass_to_halo_radius(mass)
        scaled_radius = r/rvir

        mc_vr_seed43a = self.nfw.mc_radial_velocity(scaled_radius, mass, carr, seed=43)
        mc_vr_seed43b = self.nfw.mc_radial_velocity(scaled_radius, mass, carr, seed=43)
        mc_vr_seed44 = self.nfw.mc_radial_velocity(scaled_radius, mass, carr, seed=44)

        assert np.allclose(mc_vr_seed43a, mc_vr_seed43b, rtol=0.001)
        assert not np.allclose(mc_vr_seed43a, mc_vr_seed44, rtol=0.001)

    def test_mc_pos1(self):
        """ Verify that the seed keyword is treated properly.
        This function serves as a regression test for https://github.com/astropy/halotools/issues/672.
        """
        nfw = NFWPhaseSpace()
        halos = Table()
        num_halos = int(1e4)
        halos['x'] = np.zeros(num_halos)
        halos['y'] = np.zeros(num_halos)
        halos['z'] = np.zeros(num_halos)
        halos['host_centric_distance'] = 0.
        halos['halo_rvir'] = 1.
        halos['conc_NFWmodel'] = 5.
        halos['halo_mvir'] = 1e12
        nfw.mc_pos(table=halos, seed=43)

        assert np.min(halos['x']) < -0.9
        assert np.min(halos['y']) < -0.9
        assert np.min(halos['z']) < -0.9
        assert np.max(halos['x']) > 0.9
        assert np.max(halos['y']) > 0.9
        assert np.max(halos['z']) > 0.9

    def test_mc_vel1(self):
        """
        """
        halo_table = deepcopy(self._dummy_halo_table)
        assert np.all(halo_table['vx'] == halo_table['halo_vx'])
        self.nfw.mc_vel(halo_table, seed=fixed_seed)
        assert np.any(halo_table['vx'] != halo_table['halo_vx'])

    def test_mc_vel2(self):
        """ Method verifies that seed keyword is treated properly.
        This serves as a regression test for https://github.com/astropy/halotools/issues/672
        """
        nfw = NFWPhaseSpace()
        halos = Table()
        num_halos = int(1e4)
        halos['vx'] = np.zeros(num_halos)
        halos['vy'] = np.zeros(num_halos)
        halos['vz'] = np.zeros(num_halos)
        halos['host_centric_distance'] = np.random.uniform(0, 1, num_halos)
        halos['halo_rvir'] = 1.
        halos['conc_NFWmodel'] = 5.
        halos['halo_mvir'] = 1e12
        nfw.mc_vel(halos, seed=43)
        assert not np.all(halos['vx'] == halos['vy'])
        assert not np.all(halos['vx'] == halos['vz'])

    def test_seed_treatment1(self):
        """ Regression test for https://github.com/astropy/halotools/issues/672.
        """
        nfw = NFWPhaseSpace()
        satellites = nfw.mc_generate_nfw_phase_space_points(seed=43, Ngals=100)
        assert np.any(satellites['vx'] != satellites['vy'])

    def tearDown(self):
        del self._dummy_halo_table
