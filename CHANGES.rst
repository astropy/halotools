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
