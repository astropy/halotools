"""
"""
from .....sim_manager import FakeSim
from ....factories import PrebuiltHodModelFactory
import numpy as np


def test_mass_definition_flexibility2():
    """Regression test for Issue #993."""
    halocat = FakeSim()
    halocat.halo_table["halo_mass_custom"] = np.copy(halocat.halo_table["halo_mvir"])
    halocat.halo_table["halo_radius_custom"] = np.copy(halocat.halo_table["halo_rvir"])
    halocat.halo_table.remove_column("halo_upid")
    halocat.halo_table.remove_column("halo_mvir")
    halocat.halo_table.remove_column("halo_rvir")
    model = PrebuiltHodModelFactory(
        "zheng07",
        mdef="custom",
        halo_mass_column_key="halo_mass_custom",
        prim_haloprop_key="halo_mass_custom",
        halo_boundary_key="halo_radius_custom",
    )
    assert model.halo_boundary_key == "halo_mass_custom"
    assert model.prim_haloprop_key == "halo_mass_custom"
    model.populate_mock(halocat)
