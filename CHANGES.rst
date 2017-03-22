0.5 (unreleased)
----------------

- Added SubhaloPhaseSpace class for modeling HOD-style satellite velocities with subhalos.

- The ``Lbox`` attribute of all Halotools halo catalogs is now stored as an array of length 3 instead of a scalar.

- Fixed bug in multi-threaded calculations using npairs_per_object_3d function.

- Added `empirical_models.abunmatch` sub-package providing conditional abundance matching functionality

- Fixed bug in mock population when using the seed keyword, resolving Issue #672, https://github.com/astropy/halotools/issues/672.

- Added `mock_observables.hod_from_mock` convenience function to calculate the HOD from a mock galaxy sample

- Performance enhancements to marked correlation functions

- Fixed bug in `mock_observables.pair_counters.npairs_s_mu` that impacted `mock_observables.s_mu_tpcf`

- Added Zu & Mandelbaum 2015 and Zu & Mandelbaum 2016 HOD models

- Modified internals of `mock_observables.delta_sigma` function, including an API change by removing the ``pi_max`` argument. Additionally included new `mock_observables.delta_sigma_from_precomputed` function to compute the results from a set of pre-computed pairs. See https://github.com/astropy/halotools/pull/696.

- Fixed factor of 2 error in tpcf_multipole, resolving https://github.com/astropy/halotools/issues/651

- Complete refactoring of the halotools/empirical_models/phase_space_models. No changes to either the API or behavior of any associated classes or functions.

- Addition of two new classes `halotools.empirical_models.BiasedNFWPhaseSpace` and `halotools.empirical_models.SFRBiasedNFWPhaseSpace` for NFW satellites with biased concentrations and Jeans solutions.

0.4 (2016-08-11)
----------------

- All models now support an optional ``seed`` keyword argument, allowing for deterministic Monte Carlo realizations of models. As a result of this feature, it is now mandatory that all user-defined models obey a new constraint. Any function appearing in the ``mock_generation_calling_sequence`` must now use the kwargs Python syntax to catch any additional inputs passed to these functions by the MockFactory.

- Added relative_positions_and_velocities function to mock_observables

- Fixed little h bug in the Behroozi10SmHm class. This impacts the Leauthaud11Cens and Leauthaud11Sats classes, as well as the `leauthaud11` composite model.

- Fixed bug in mock_observables.pair_counters.npairs_per_object_3d. See https://github.com/astropy/halotools/issues/606.

- New counts_in_cells sub-package in mock_observables

- HodMockFactory has new estimate_ngals feature

- Fixed buggy behavior for two-point functions called for logical branch `do_auto=True, do_cross=False`

- Performance enhancement of isolation_functions by 50% - 300%, depending on numerical regime.

- Updated all catalogs to version_name=``halotools_v0p4``, resolving the bug pointed out in https://github.com/astropy/halotools/issues/598.

- Performance enhancement of npairs_s_mu function by 10-100x after cleaning cython engine of python objects.


0.3 (2016-06-28)
----------------

- Removed distant_observer_redshift function from mock_survey module

- Removed -march=native compiler flag to resolve installation problems on some architectures


0.2 (2016-06-09)
----------------

- Halotools is now Python 3.x compatible

- Halotools mock_observables package has been given a complete overhaul, adding many new cythonized engines and pair counters (listed below). Functions are 30% - 50x faster, depending on numerical regime. Overhauled engines include velocity_marked_npairs_3d, velocity_marked_npairs_xy_z, npairs_per_object_3d, npairs_s_mu, npairs_jackknife_3d, npairs_projected, npairs_xy_z, npairs_3d, marked_npairs_3d and marked_npairs_xy_z

- Added new utils.crossmatch function

- Added new mock_observables.radial_profile_3d function

- All isolation_functions now return boolean ``is_isolated`` rather than its inverse ``has_neighbor``

- Fixed a bug in mock_observables.delta_sigma. See https://github.com/astropy/halotools/issues/523

- Fixed bug in mock_observables.tpcf_jackknife. See https://github.com/astropy/halotools/issues/513

- Deleted mock_observables.nearest_neighbor function


0.1 (2016-03-13)
----------------

- Initial release
