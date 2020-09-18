"""
"""
import pytest
import numpy as np

from ....occupation_models import OccupationComponent, Zheng07Cens, Zheng07Sats
from ....phase_space_models import TrivialPhaseSpace, NFWPhaseSpace
from ....factories import PrebuiltHodModelFactory, HodModelFactory

from .....custom_exceptions import HalotoolsError
from .....sim_manager import FakeSim

__all__ = ("test_zheng07_composite1", "test_zheng07_composite2")


def test_zheng07_composite1():
    """ Ensure that the ``zheng07`` pre-built model does not accept non-Zheng07Cens.
    """
    model1 = PrebuiltHodModelFactory("zheng07")
    model2 = PrebuiltHodModelFactory("zheng07", modulate_with_cenocc=True)

    with pytest.raises(HalotoolsError) as err:
        __ = PrebuiltHodModelFactory(
            "zheng07", modulate_with_cenocc=True, cenocc_model=0
        )
    substr = (
        "Do not pass in the ``cenocc_model`` keyword to ``zheng07_model_dictionary``"
    )
    assert substr in err.value.args[0]


def test_zheng07_composite2():
    """ This test ensures that the source code provided in the
    ``Advanced usage of the ``zheng07`` model`` tutorial behaves as expected.
    """

    class MyCenModel(OccupationComponent):
        def __init__(self, threshold):
            OccupationComponent.__init__(
                self,
                gal_type="centrals",
                threshold=threshold,
                upper_occupation_bound=1.0,
            )

            self.param_dict["new_cen_param"] = 0.5
            self._suppress_repeated_param_warning = True

        def mean_occupation(self, **kwargs):
            halo_table = kwargs["table"]
            result = np.zeros(len(halo_table)) + self.param_dict["new_cen_param"]
            return result

    centrals_occupation = MyCenModel(threshold=-20)
    satellites_occupation = Zheng07Sats(
        threshold=-20,
        modulate_with_cenocc=True,
        cenocc_model=centrals_occupation,
        _suppress_repeated_param_warning=True,
    )

    centrals_profile = TrivialPhaseSpace()
    satellites_profile = NFWPhaseSpace()

    model_dict = {
        "centrals_occupation": centrals_occupation,
        "centrals_profile": centrals_profile,
        "satellites_occupation": satellites_occupation,
        "satellites_profile": satellites_profile,
    }

    composite_model = HodModelFactory(**model_dict)

    fake_sim = FakeSim()
    composite_model.populate_mock(fake_sim)


def test_modulate_with_cenocc1():
    """ Regression test for Issue #646. Verify that the ``modulate_with_cenocc``
    keyword argument is actually passed to the satellite occupation component.
    """
    model1 = PrebuiltHodModelFactory("zheng07", modulate_with_cenocc=True)
    assert (
        model1.model_dictionary[u"satellites_occupation"].modulate_with_cenocc is True
    )


def test_modulate_with_cenocc2():
    """ Regression test for Issue #646. Verify that the ``modulate_with_cenocc``
    keyword argument results in behavior that is properly modified by the centrals.
    """
    model = PrebuiltHodModelFactory("zheng07", modulate_with_cenocc=True)
    test_mass = 1e12
    ncen1 = model.mean_occupation_satellites(prim_haloprop=test_mass)
    model.param_dict["logMmin"] *= 1.5
    ncen2 = model.mean_occupation_satellites(prim_haloprop=test_mass)
    assert ncen1 != ncen2


def test_host_centric_distance():
    model = PrebuiltHodModelFactory("zheng07")
    halocat = FakeSim(seed=43)
    model.populate_mock(halocat, seed=43)

    msg = "host_centric_distance key was never mapped during mock population"
    assert np.any(model.mock.galaxy_table["host_centric_distance"] > 0), msg


def test_mass_definition_flexibility():
    """ Regression test for Issue #993.
    """
    model = PrebuiltHodModelFactory(
        "zheng07", mdef="200m", halo_boundary_key="halo_radius_arbitrary"
    )
    halocat = FakeSim(seed=43)
    halocat.halo_table["halo_m200m"] = np.copy(halocat.halo_table["halo_mvir"])
    halocat.halo_table["halo_radius_arbitrary"] = np.copy(
        halocat.halo_table["halo_mvir"]
    )
    model.populate_mock(halocat, seed=43)
