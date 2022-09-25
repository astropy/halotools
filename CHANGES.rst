0.8 (2022-09-25)
----------------

- Change packaging tools to pyproject.toml

- Drop dependency on astropy-helpers

- Add new calculator for uber_hostid quantity

- Bug fix for cross-correlations with the Hamilton and Landy-Szalay estimators. See https://github.com/astropy/halotools/pull/1032

- Bug fix for tpcf jackknife. See https://github.com/astropy/halotools/pull/1020

- Bug fix for the central occupation function of the Zu & Mandelbaum 2016 HOD. See https://github.com/astropy/halotools/pull/1028.

- Add support for non-Poissonian satellite occupation statistics. See https://github.com/astropy/halotools/pull/996

- Add vector rotation utilities to halotools.utils

- Add support for Conditional counts in cylinders. See https://github.com/astropy/halotools/pull/987

0.7 (2020-1-29)
----------------

- Added new `matrix_operations_3d` module in the `utils` subpackage that contains Numpy functions related to three-dimensional rotations.

- Added new `inertia_tensor_per_object` function in `mock_observables` that calculates the inertia tensor for a set of points and masses in a periodic box.

- Added new option for the marked correlation functions to accommodate counting pairs of points passing a variable merger ratio criteria

- Added `~halotools.utils.fuzzy_digitize` function to `~halotools.utils` sub-package.

- Added `~halotools.utils.sliding_conditional_percentile` function to `~halotools.utils` sub-package.

- Added new `resample_x_to_match_y` function to `halotools.utils`.

- Renamed old implementation of `conditional_abunmatch` to `conditional_abunmatch_bin_based`

- Added new bin-free implementation of `conditional_abunmatch`.

- Added new utils function `bijective_distribution_matching`

- Added new `load_um_binary_sfr_catalog` function to load SFR catalogs from UniverseMachine into memory

- Added new `return_indexes` feature to conditional_abunmatch function. See https://github.com/astropy/halotools/pull/913.

- New function `mean_delta_sigma` replaces old `delta_sigma` function. See #955.


0.6 (2017-12-15)
----------------

- Changed the API for mock_observables.pair_counters.npairs_s_mu which now requires ``mu_bins`` to be in the conventional mu=cos(theta_LOS) format instead of mu=sin(theta_LOS). See https://github.com/astropy/halotools/pull/768

- Added new `mock_observables` functions `radial_distance` and `radial_distance_and_velocity` functions. See https://github.com/astropy/halotools/pull/782

- Added new `distribution_matching_indices` function. See https://github.com/astropy/halotools/pull/795.

- Removed `max_sample_size` keyword argument from all `mock_observables` functions.

- Added new `weighted_npairs_s_mu` function to `mock_observables` sub-package. See Issue #810

- Added standalone function to apply redshift-space distortions. See Issue #806.

- Added `num_lines_header` optional keyword argument to TabularAsciiReader

- Added `wp_jackknife` function. See https://github.com/astropy/halotools/pull/814.

- Fixed bug in the normalization of the covariance matrices in the `tpcf_jackknife` function.  See https://github.com/astropy/halotools/issues/815.

- Added `rp_pi_tpcf_jackknife` function - See #822

- Fixed bug in estimate_ngals method for case of assembly-biased occupation models. See https://github.com/astropy/halotools/issues/801

- Removed obsolete bounds_enforcing_decorator_factory function - see https://github.com/astropy/halotools/issues/756

- Fixed bug impacting ability of python 3.6 users to access downsampled particle data - See Issues #821, #826, and #831.

- Fixed bug in isolation criteria for calculations without PBCs. See Issue #776.

- Fixed bug in `weighted_npairs_s_mu` when called in parallel. See https://github.com/astropy/halotools/issues/837.

- Fixed bug in `radial_profile_3d` when called in parallel. See https://github.com/astropy/halotools/issues/854.

- Fixed bug in `radial_profile_3d` when called without periodic boundary conditions. See https://github.com/astropy/halotools/issues/862

- Fixed bug in `velocity_marked_npairs_3d` and `velocity_marked_npairs_xy_z` when called with default arguments. See https://github.com/astropy/halotools/issues/836.

- Changed the `weight_func_id` numbers associated with weighting functions for `velocity_marked_npairs_3d` and `velocity_marked_npairs_xy_z` where e.g. 11 is now 1, etc.

- Added new `test_installation` feature that dramatically shortens the length of time users need to spend verifying their installation


0.5 (2017-05-31)
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

- The ``gal_types`` attribute of HODModelFactory-produced models is now sorted so that ``centrals`` always appears before ``satellites``. This new default behavior is more common for interdependent occupation models, where satellite abundance depends upon central galaxy characteristics, rather than the other way around. See https://github.com/astropy/halotools/pull/729

- Added new keyword arguments to `return_xyz_formatted_array` function enabling application of redshift-space distortions for galaxy samples at higher redshift. Previously, the user needed to do this manually). Default behavior of this function is unchanged, provided users had not locally modified the `sim_defaults` module to have set `default_redshift` greater than zero.


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
