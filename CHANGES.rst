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
