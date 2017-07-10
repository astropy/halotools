"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from astropy.table import Table

from ..subhalo_phase_space import SubhaloPhaseSpace

from .....sim_manager import FakeSim, CachedHaloCatalog
from .....utils import crossmatch

__all__ = ('test_preprocess_subhalo_table1', )

try:
    halocat = CachedHaloCatalog()
    HAS_DEFAULT_CATALOG = True
except:
    HAS_DEFAULT_CATALOG = False

fixed_seed = 43


def test_init():
    model = SubhaloPhaseSpace('satellites', np.logspace(10, 15, 25))
    assert list(model._additional_kwargs_dict.keys()) == ['inherit_subhalo_properties']
    assert model._mock_generation_calling_sequence == ['inherit_subhalo_properties']


def test_default_inherited_subhalo_props_dict():
    from .....empirical_models import default_inherited_subhalo_props_dict
    msg = ("The ``default_inherited_subhalo_props_dict`` should be importable from empirical_models\n"
        "and it should have the ``{0}`` key")
    for key in ('halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz'):
        assert key in list(default_inherited_subhalo_props_dict.keys()), msg.format(key)


def test_preprocess_subhalo_table1():
    halocat = FakeSim(seed=fixed_seed)

    subhalo_mask = halocat.halo_table['halo_upid'] != -1
    subhalo_table = halocat.halo_table[subhalo_mask]
    host_halo_table = halocat.halo_table[~subhalo_mask]

    model = SubhaloPhaseSpace('satellites', np.logspace(10.1, 16.1, 25))
    with pytest.raises(ValueError) as err:
        hosts, subs = model.preprocess_subhalo_table(host_halo_table, subhalo_table)
    substr = "The model ``host_haloprop_bins`` spans the range"
    assert substr in err.value.args[0]


def test_preprocess_subhalo_table2():
    halocat = FakeSim(seed=fixed_seed)

    subhalo_mask = halocat.halo_table['halo_upid'] != -1
    subhalo_table = halocat.halo_table[subhalo_mask]
    host_halo_table = halocat.halo_table[~subhalo_mask]

    subhalo_table['halo_mvir_host_halo'] = 0.5

    model = SubhaloPhaseSpace('satellites', np.logspace(9.9, 16.1, 25))

    with pytest.raises(ValueError) as err:
        hosts, subs = model.preprocess_subhalo_table(host_halo_table, subhalo_table)
    substr = "The ``halo_mvir_host_halo`` column of the input ``subhalo_table``"
    assert substr in err.value.args[0]


def test_preprocess_subhalo_table3():
    halocat = FakeSim(seed=fixed_seed)

    subhalo_mask = halocat.halo_table['halo_upid'] != -1
    subhalo_table = halocat.halo_table[subhalo_mask][0:4]
    host_halo_table = halocat.halo_table[~subhalo_mask]

    model = SubhaloPhaseSpace('satellites', np.logspace(9.9, 16.1, 25))

    with pytest.raises(ValueError) as err:
        hosts, subs = model.preprocess_subhalo_table(host_halo_table, subhalo_table)
    substr = "There must be at least 1 subhalo in each bin of"
    assert substr in err.value.args[0]


def test_preprocess_subhalo_table4():
    halocat = FakeSim(seed=fixed_seed)

    subhalo_mask = halocat.halo_table['halo_upid'] != -1
    subhalo_table = halocat.halo_table[subhalo_mask]
    host_halo_table = halocat.halo_table[~subhalo_mask]

    model = SubhaloPhaseSpace('satellites', np.logspace(9.9, 16.1, 10))
    hosts, subs = model.preprocess_subhalo_table(host_halo_table, subhalo_table)


def test_retrieve_subhalo_indices1():
    halocat = FakeSim(seed=fixed_seed)

    subhalo_mask = halocat.halo_table['halo_upid'] != -1
    subhalo_table = halocat.halo_table[subhalo_mask]
    host_halo_table = halocat.halo_table[~subhalo_mask]

    model = SubhaloPhaseSpace('satellites', np.logspace(9.9, 16.1, 10))
    hosts, subs = model.preprocess_subhalo_table(host_halo_table, subhalo_table)

    with NumpyRNGContext(fixed_seed):
        occupations = np.random.randint(0, 3, len(hosts))

    idx, missing_subhalo_mask = model._retrieve_satellite_selection_idx(
        hosts, subs, occupations, seed=fixed_seed)

    try:
        selected_subs = subs[idx]
    except IndexError:
        msg = ("Output of ``_retrieve_satellite_selection_idx`` function fails to \n"
            "serve as an indexing array into the pre-processed ``subhalo_table``.\n")
        raise ValueError(msg)

    assert len(selected_subs) == occupations.sum()


@pytest.mark.skipif('not HAS_DEFAULT_CATALOG')
def test_retrieve_subhalo_indices2():
    halocat = CachedHaloCatalog()

    mass_cut = 3000*halocat.particle_mass
    initial_cut = halocat.halo_table['halo_mvir'] > mass_cut
    halos = halocat.halo_table[initial_cut]

    subhalo_mask = halos['halo_upid'] != -1
    subhalo_table = halos[subhalo_mask]
    host_halo_table = halos[~subhalo_mask]

    log10_mmin = np.log10(mass_cut)-0.1
    log10_mmax = np.log10(halos['halo_mpeak'].max())+0.1
    num_mass_bins = 10
    mass_bins = np.logspace(log10_mmin, log10_mmax, num_mass_bins)
    model = SubhaloPhaseSpace('satellites', mass_bins)

    hosts, subs = model.preprocess_subhalo_table(host_halo_table, subhalo_table)

    with NumpyRNGContext(fixed_seed):
        occupations = np.random.randint(0, 3, len(hosts))

    idx, missing_subhalo_mask = model._retrieve_satellite_selection_idx(
        hosts, subs, occupations, seed=fixed_seed)

    try:
        selected_subs = subs[idx]
    except IndexError:
        msg = ("Output of ``_retrieve_satellite_selection_idx`` function fails to \n"
            "serve as an indexing array into the pre-processed ``subhalo_table``.\n")
        raise ValueError(msg)

    assert len(selected_subs) == occupations.sum()

    idx2, missing_subhalo_mask = model._retrieve_satellite_selection_idx(
        hosts, subs, occupations, seed=fixed_seed)
    assert np.all(idx == idx2)

    idx3, missing_subhalo_mask = model._retrieve_satellite_selection_idx(
        hosts, subs, occupations, seed=fixed_seed+1)

    assert np.any(missing_subhalo_mask == True)
    assert not np.all(idx == idx3)
    assert np.all(idx[~missing_subhalo_mask] == idx3[~missing_subhalo_mask])


def test_inherit_subhalo_properties():
    halocat = FakeSim(seed=fixed_seed, num_halos_per_massbin=1000)

    subhalo_mask = halocat.halo_table['halo_upid'] != -1
    subhalo_table = halocat.halo_table[subhalo_mask]
    host_halo_table = halocat.halo_table[~subhalo_mask]

    model = SubhaloPhaseSpace('satellites', np.logspace(9.9, 16.1, 10))
    hosts, subs = model.preprocess_subhalo_table(host_halo_table, subhalo_table)

    with NumpyRNGContext(fixed_seed):
        occupations = np.random.randint(0, 2, len(hosts))

    galaxy_table = Table()
    galaxy_table['real_subhalo'] = np.zeros(occupations.sum(), dtype=np.dtype(bool))
    for item in model.inherited_subhalo_props_dict.values():
        key, dt = item[0], np.dtype(item[1])
        galaxy_table[key] = np.zeros(occupations.sum(), dtype=dt)

    assert np.all(galaxy_table['x'] == 0)

    _occupations = {'satellites': occupations}
    model.inherit_subhalo_properties(table=galaxy_table,
        halo_table=hosts, subhalo_table=subs, _occupation=_occupations,
        seed=fixed_seed, Lbox=halocat.Lbox)

    msg = "This fails because _inherit_props_for_remaining_satellites is still unfinished"
    zero_mask = galaxy_table['x'] == 0
    fake_mask = galaxy_table['real_subhalo'] == True
    print(len(galaxy_table), len(galaxy_table[zero_mask]), len(galaxy_table[fake_mask]),
        set(galaxy_table['real_subhalo'][zero_mask]))
    assert not np.any(galaxy_table['x'] == 0), msg

    idxA, idxB = crossmatch(galaxy_table['halo_id'].data, subs['halo_id'].data)
    assert len(galaxy_table['halo_id'][idxA]) == len(galaxy_table['halo_id'])
