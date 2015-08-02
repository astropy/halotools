from time import time
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


from halotools.utils.table_utils import compute_conditional_percentiles
hosts['halo_nfw_conc_percentile'] = compute_conditional_percentiles(
    halo_table = hosts, 
    prim_haloprop_key = abz.prim_haloprop_key, 
    sec_haloprop_key = abz.sec_haloprop_key
    )

def master_func():
	return abz.mean_occupation(halo_table = hosts)

# occ = prof.runcall(master_func)
# prof.print_stats(sort='tottime')
# prof.dump_stats('profile_abz.prof')

print("\n\n")
t0 = time()
x = master_func()
t1 = time()
t = (t1 - t0)*1000.
print("\n\n\n\n Total runtime of function call = %.2f ms\n\n\n\n" % t)







