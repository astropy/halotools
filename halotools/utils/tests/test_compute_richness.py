"""
"""
import numpy as np
from ..crossmatch import compute_richness


__all__ = ('test_compute_richness1', )


def test_compute_richness1():
    unique_halo_ids = [5, 2, 100]
    halo_id_of_galaxies = [100, 2, 100, 3, 2, 100, 100, 3]
    richness = compute_richness(unique_halo_ids, halo_id_of_galaxies)
    assert np.all(richness == [0, 2, 4])


def test_compute_richness2():
    unique_halo_ids = [400, 100, 200, 300]
    halo_id_of_galaxies = np.random.randint(0, 50, 200)
    richness = compute_richness(unique_halo_ids, halo_id_of_galaxies)
    assert np.all(richness == [0, 0, 0, 0])


def test_compute_richness3():
    unique_halo_ids = [400, 100, 200, 300]
    halo_id_of_galaxies = [0, 999, 100, 200, 100, 200, 999, 300, 200]
    richness = compute_richness(unique_halo_ids, halo_id_of_galaxies)
    assert np.all(richness == [0, 2, 3, 1])


def test_compute_richness4():
    #  Set up a source halo catalog with 100 halos in each mass bin
    log_mhost_min, log_mhost_max, dlog_mhost = 10.5, 15.5, 0.5
    log_mhost_bins = np.arange(log_mhost_min, log_mhost_max+dlog_mhost, dlog_mhost)
    log_mhost_mids = 0.5*(log_mhost_bins[:-1] + log_mhost_bins[1:])

    num_source_halos_per_bin = 100
    source_halo_log_mhost = np.tile(log_mhost_mids, num_source_halos_per_bin)
    num_source_halos = len(source_halo_log_mhost)
    source_halo_id = np.arange(num_source_halos).astype(int)

    ngals_per_source_halo = 3
    num_source_galaxies = num_source_halos*ngals_per_source_halo
    source_galaxy_host_halo_id = np.repeat(source_halo_id, ngals_per_source_halo)

    source_halos_richness = compute_richness(
                source_halo_id, source_galaxy_host_halo_id)





