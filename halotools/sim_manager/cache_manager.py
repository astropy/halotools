
import os
try:
    import h5py
except ImportError:
    warn("Most of the functionality of the "
        "sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda. ")

from .halo_table_cache import HaloTableCache

class CacheManager(object):
    """ Object providing a collection of halo catalogs for use with Halotools. 
    """ 

    def __init__(self):
        """
        """
        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py package installed "
                "to use the CacheManager")

        self.halo_table_cache = HaloTableCache()

    def matching_halo_tables(self, **kwargs):
        """
        """

        return list(self.halo_table_cache.matching_log_entry_generator(**kwargs))




