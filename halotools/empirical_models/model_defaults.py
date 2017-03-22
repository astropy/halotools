"""
Module expressing various default settings of the empirical modeling sub-package.

"""
import numpy as np

__all__ = ['get_halo_boundary_key', 'get_halo_mass_key']


# Default thresholds for mocks
default_luminosity_threshold = -20
default_stellar_mass_threshold = 10.5

# Small numerical value passed to the scipy Poisson number generator.
# Used when executing a Monte Carlo realization of a Poission distribution
# whose mean is formally zero, which causes the built-in
# scipy method to raise an exception.
default_tiny_poisson_fluctuation = 1.e-20

default_smhm_scatter = 0.2
default_smhm_haloprop = 'halo_mpeak'
default_binary_galprop_haloprop = default_smhm_haloprop

# At minimum, the following halo and galaxy properties
# will be bound to each mock galaxy
host_haloprop_prefix = 'halo_'

default_haloprop_list_inherited_by_mock = (
    ['halo_id', 'halo_hostid', 'halo_x', 'halo_y', 'halo_z',
    'halo_vx', 'halo_vy', 'halo_vz',
    'halo_mvir', 'halo_rvir', 'halo_upid']
    )

prim_haloprop_key = 'halo_mvir'
sec_haloprop_key = 'halo_nfw_conc'

halo_mass_definition = 'vir'


def get_halo_boundary_key(mdef):
    """ For the input mass definition,
    return the string used to access halo table column
    storing the halo radius.

    For example, the function will return ``halo_rvir`` if passed the string ``vir``,
    and will return ``halo_r200m`` if passed ``200m``, each of which correspond to the
    Halotools convention for the column storing the distance between the host halo center
    and host halo boundary in `~halotools.sim_manager.CachedHaloCatalog` data tables.

    Parameters
    -----------
    mdef: str
        String specifying the halo mass definition, e.g., 'vir' or '200m'.

    Returns
    --------
    radius_key : str
    """
    return 'halo_r'+mdef


def get_halo_mass_key(mdef):
    """ For the input mass definition,
    return the string used to access halo table column
    storing the halo mass.

    For example, the function will return ``halo_mvir`` if passed the string ``vir``,
    and will return ``halo_m200m`` if passed ``200m``, each of which correspond to the
    Halotools convention for the column storing the halo mass in
    `~halotools.sim_manager.CachedHaloCatalog` data tables.

    Parameters
    -----------
    mdef: str
        String specifying the halo mass definition, e.g., 'vir' or '200m'.

    Returns
    --------
    mass_key : str
    """
    return 'halo_m'+mdef


# Number of bins to use in the lookup table attached to the NFWProfile.
# Used primarily by HODMockFactory.

min_permitted_conc = 2.0
max_permitted_conc = 20.0

default_conc_gal_bias_bins = np.linspace(0.1, 10, 10)
default_conc_gal_bias_bins = np.insert(default_conc_gal_bias_bins,
    np.searchsorted(default_conc_gal_bias_bins, 1), 1)

Npts_radius_table = 101
default_lograd_min = -3
default_lograd_max = 0
conc_mass_model = 'direct_from_halo_catalog'
concentration_key = 'halo_nfw_conc'


default_rbins = np.logspace(-1, 1.25, 15)
default_nptcls = 1e5

default_b_perp = 0.2
default_b_para = 0.75
