import pstats, cProfile
prof = cProfile.Profile()

import numpy as np
from halotools.empirical_models import assembias
reload(assembias)

from halotools.sim_manager import HaloCatalog
halocat = HaloCatalog()
halos = halocat.halo_table
host_mask = halos['halo_hostid'] == halos['halo_id']
hosts = halos[host_mask]

abz = assembias.AssembiasZheng07Cens()

def master_func():
	return abz.mean_occupation(halo_table = hosts)

# occ = prof.runcall(master_func)
# prof.print_stats(sort='tottime')
# prof.dump_stats('profile_abz.prof')

x = master_func()








