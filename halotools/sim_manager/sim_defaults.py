"""
Module expressing various default settings of the simulation manager sub-package.

All values hard-coded here appear as unique variables throughout the entire Halotools code base.
This allows you to customize your default settings and be guaranteed that whatever changes you make
will correctly propagate to all relevant behavior. See the in-line comments in the
``halotools/sim_manager/sim_defaults.py`` source code for
descriptions of the purpose of each variable defined in this module.

"""

from astropy import cosmology

# Set the default argument for the CachedHaloCatalog class.
# Any combination of simname, halo-finder and redshift may be chosen,
# provided that the selected combination corresponds to a catalog in your cache directory
# These choices dictate any behavior that loads an unspecified CachedHaloCatalog into memory,
# such as calling the `populate_mock()` method of composite models
default_simname = 'bolshoi'
default_halo_finder = 'rockstar'
default_version_name = 'halotools_v0p4'
default_redshift = 0.0
default_ptcl_version_name = 'halotools_v0p4'

# The following two variables are used to define completeness cuts applied to halo catalogs
# The Halotools default completeness cut is to throw out all halos with mpeak < mp*300,
# that is, to throw out any halo for which the virial mass of the main progenitor
# never exceeded 300 particles at any point in its assembly history
# These two variables control how the initially-downloaded "raw" subhalo catalogs were processed
# and converted to the hdf5 binaries made available with Halotools.
# If you wish to use a different completeness definition, you can either modify these two variables,
# or alternatively use a custom-defined cut on the raw catalog.
Num_ptcl_requirement = 300
mass_like_variable_to_apply_cut = 'halo_mpeak'

default_cosmology = cosmology.WMAP5

# URLs of websites hosting catalogs used by the package
processed_halo_tables_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/halo_catalogs'
ptcl_tables_webloc = 'http://www.astro.yale.edu/aphearin/Data_files/particle_catalogs'

default_cache_location = 'pkg_default'


############################################################
