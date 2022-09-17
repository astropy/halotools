"""
"""
import numpy as np

import pytest
from astropy.utils.misc import NumpyRNGContext

from unittest import TestCase
import warnings

from ...sim_manager import FakeSim

__all__ = ["TestHodModelFactoryTutorial"]

fixed_seed = 43


class TestHodModelFactoryTutorial(TestCase):
    """ """

    def test_hod_modeling_tutorial1(self):

        from ...empirical_models import HodModelFactory

        from ...empirical_models import TrivialPhaseSpace, Zheng07Cens

        cens_occ_model = Zheng07Cens()
        cens_prof_model = TrivialPhaseSpace()

        from ...empirical_models import NFWPhaseSpace, Zheng07Sats

        sats_occ_model = Zheng07Sats()
        sats_prof_model = NFWPhaseSpace()

        model_instance = HodModelFactory(
            centrals_occupation=cens_occ_model,
            centrals_profile=cens_prof_model,
            satellites_occupation=sats_occ_model,
            satellites_profile=sats_prof_model,
        )

        # The model_instance is a composite model
        # All composite models can directly populate N-body simulations
        # with mock galaxy catalogs using the populate_mock method:
        halocat = FakeSim()
        model_instance.populate_mock(halocat)

        # Setting simname to 'fake' populates a mock into a fake halo catalog
        # that is generated on-the-fly, but you can use the populate_mock
        # method with any Halotools-formatted catalog

    def test_hod_modeling_tutorial2a(self):
        from ...empirical_models import HodModelFactory

        from ...empirical_models import TrivialPhaseSpace, Zheng07Cens

        another_cens_occ_model = Zheng07Cens()
        another_cens_prof_model = TrivialPhaseSpace()

        from ...empirical_models import NFWPhaseSpace, Zheng07Sats

        another_sats_occ_model = Zheng07Sats()
        another_sats_prof_model = NFWPhaseSpace()

        from ...empirical_models import HaloMassInterpolQuenching

        sat_quenching = HaloMassInterpolQuenching(
            "halo_mvir",
            [1e12, 1e13, 1e14, 1e15],
            [0.35, 0.5, 0.6, 0.9],
            gal_type="satellites",
        )
        cen_quenching = HaloMassInterpolQuenching(
            "halo_mvir", [1e12, 1e15], [0.25, 0.95], gal_type="centrals"
        )

        model_instance = HodModelFactory(
            centrals_occupation=another_cens_occ_model,
            centrals_profile=another_cens_prof_model,
            satellites_occupation=another_sats_occ_model,
            satellites_profile=another_sats_prof_model,
            centrals_quenching=cen_quenching,
            satellites_quenching=sat_quenching,
        )

    def test_hod_modeling_tutorial2b(self):

        from ...empirical_models import HodModelFactory

        from ...empirical_models import TrivialPhaseSpace, Leauthaud11Cens

        another_cens_occ_model = Leauthaud11Cens()
        another_cens_prof_model = TrivialPhaseSpace()

        from ...empirical_models import NFWPhaseSpace, Leauthaud11Sats

        another_sats_occ_model = Leauthaud11Sats()
        another_sats_prof_model = NFWPhaseSpace()

        from ...empirical_models import HaloMassInterpolQuenching

        sat_quenching = HaloMassInterpolQuenching(
            "halo_mvir",
            [1e12, 1e13, 1e14, 1e15],
            [0.35, 0.5, 0.6, 0.9],
            gal_type="satellites",
        )
        cen_quenching = HaloMassInterpolQuenching(
            "halo_mvir", [1e12, 1e15], [0.25, 0.95], gal_type="centrals"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model_instance = HodModelFactory(
                centrals_occupation=another_cens_occ_model,
                centrals_profile=another_cens_prof_model,
                satellites_occupation=another_sats_occ_model,
                satellites_profile=another_sats_prof_model,
                centrals_quenching=cen_quenching,
                satellites_quenching=sat_quenching,
            )
            assert len(w) > 0
            assert "appears in more than one component model" in str(w[-1].message)

        cen_quenching._suppress_repeated_param_warning = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model_instance = HodModelFactory(
                centrals_occupation=another_cens_occ_model,
                centrals_profile=another_cens_prof_model,
                satellites_occupation=another_sats_occ_model,
                satellites_profile=another_sats_prof_model,
                centrals_quenching=cen_quenching,
                satellites_quenching=sat_quenching,
            )
            assert len(w) == 0

        assert hasattr(model_instance, "mean_quiescent_fraction_centrals")
        assert hasattr(model_instance, "mean_quiescent_fraction_satellites")

        assert "centrals_quiescent_ordinates_param1" in list(
            model_instance.param_dict.keys()
        )
        assert "centrals_quiescent_ordinates_param2" in list(
            model_instance.param_dict.keys()
        )
        assert "satellites_quiescent_ordinates_param1" in list(
            model_instance.param_dict.keys()
        )
        assert "satellites_quiescent_ordinates_param4" in list(
            model_instance.param_dict.keys()
        )

        halocat = FakeSim()
        model_instance.populate_mock(halocat)

        assert "quiescent" in list(model_instance.mock.galaxy_table.keys())
        assert set(model_instance.mock.galaxy_table["quiescent"]) == {True, False}

        cenmask = model_instance.mock.galaxy_table["gal_type"] == "centrals"
        cens = model_instance.mock.galaxy_table[cenmask]
        assert set(cens["quiescent"]) == {True, False}

        satmask = model_instance.mock.galaxy_table["gal_type"] == "satellites"
        sats = model_instance.mock.galaxy_table[satmask]
        assert set(sats["quiescent"]) == {True, False}

    def test_hod_modeling_tutorial2c(self):

        from ...empirical_models import HodModelFactory

        from ...empirical_models import TrivialPhaseSpace, Zheng07Cens

        another_cens_occ_model = Zheng07Cens()
        another_cens_prof_model = TrivialPhaseSpace()

        from ...empirical_models import NFWPhaseSpace, Zheng07Sats

        another_sats_occ_model = Zheng07Sats()
        another_sats_prof_model = NFWPhaseSpace()

        ordinary_zheng07_model = HodModelFactory(
            centrals_occupation=another_cens_occ_model,
            centrals_profile=another_cens_prof_model,
            satellites_occupation=another_sats_occ_model,
            satellites_profile=another_sats_prof_model,
        )

        from ...empirical_models import HaloMassInterpolQuenching

        sat_quenching = HaloMassInterpolQuenching(
            "halo_mvir",
            [1e12, 1e13, 1e14, 1e15],
            [0.35, 0.5, 0.6, 0.9],
            gal_type="satellites",
        )
        cen_quenching = HaloMassInterpolQuenching(
            "halo_mvir", [1e12, 1e15], [0.25, 0.95], gal_type="centrals"
        )

        zheng07_with_quenching = HodModelFactory(
            baseline_model_instance=ordinary_zheng07_model,
            centrals_quenching=cen_quenching,
            satellites_quenching=sat_quenching,
        )

    def test_hod_modeling_tutorial3(self):
        class Size(object):
            def __init__(self, gal_type):

                self.gal_type = gal_type
                self._mock_generation_calling_sequence = ["assign_size"]
                self._galprop_dtypes_to_allocate = np.dtype([("galsize", "f4")])
                self.list_of_haloprops_needed = ["halo_spin"]

            def assign_size(self, table, **kwargs):

                table["galsize"][:] = table["halo_spin"] / 5.0

        cen_size = Size("centrals")
        sat_size = Size("satellites")
        from ...empirical_models import PrebuiltHodModelFactory, HodModelFactory

        zheng_model = PrebuiltHodModelFactory("zheng07")
        new_model = HodModelFactory(
            baseline_model_instance=zheng_model,
            centrals_size=cen_size,
            satellites_size=sat_size,
        )

        assert hasattr(new_model, "assign_size_centrals")

        halocat = FakeSim()
        new_model.populate_mock(halocat)
        assert "galsize" in list(new_model.mock.galaxy_table.keys())
        assert len(set(new_model.mock.galaxy_table["galsize"])) > 0

    def test_hod_modeling_tutorial4(self):
        class Shape(object):
            def __init__(self, gal_type, prim_haloprop_key):

                self.gal_type = gal_type
                self._mock_generation_calling_sequence = [
                    "assign_disrupted",
                    "assign_axis_ratio",
                ]
                self._galprop_dtypes_to_allocate = np.dtype(
                    [("axis_ratio", "f4"), ("disrupted", bool)]
                )
                self.list_of_haloprops_needed = ["halo_spin"]

                self.prim_haloprop_key = prim_haloprop_key
                self._methods_to_inherit = [
                    "assign_disrupted",
                    "assign_axis_ratio",
                    "disrupted_fraction_vs_halo_mass",
                ]
                self.param_dict = {
                    "max_disruption_mass_" + self.gal_type: 1e12,
                    "disrupted_fraction_" + self.gal_type: 0.25,
                }

            def assign_disrupted(self, **kwargs):
                if "table" in list(kwargs.keys()):
                    table = kwargs["table"]
                    halo_mass = table[self.prim_haloprop_key]
                else:
                    halo_mass = kwargs["prim_haloprop"]

                disrupted_fraction = self.disrupted_fraction_vs_halo_mass(halo_mass)
                with NumpyRNGContext(fixed_seed):
                    randomizer = np.random.uniform(0, 1, len(halo_mass))
                is_disrupted = randomizer < disrupted_fraction

                if "table" in list(kwargs.keys()):
                    table["disrupted"][:] = is_disrupted
                else:
                    return is_disrupted

            def assign_axis_ratio(self, **kwargs):
                table = kwargs["table"]
                mask = table["disrupted"] == True
                num_disrupted = len(table["disrupted"][mask])
                with NumpyRNGContext(fixed_seed):
                    table["axis_ratio"][mask] = np.random.random(num_disrupted)
                table["axis_ratio"][~mask] = 0.3

            def disrupted_fraction_vs_halo_mass(self, mass):
                bool_mask = (
                    mass > self.param_dict["max_disruption_mass_" + self.gal_type]
                )
                val = self.param_dict["disrupted_fraction_" + self.gal_type]
                return np.where(bool_mask == True, 0, val)

        cen_shape = Shape("centrals", "halo_mvir")
        sat_shape = Shape("satellites", "halo_m200b")
        from ...empirical_models import PrebuiltHodModelFactory, HodModelFactory

        zheng_model = PrebuiltHodModelFactory("zheng07")
        new_model = HodModelFactory(
            baseline_model_instance=zheng_model,
            centrals_shape=cen_shape,
            satellites_shape=sat_shape,
        )

        halocat = FakeSim()
        new_model.populate_mock(halocat)
        assert "axis_ratio" in list(new_model.mock.galaxy_table.keys())
        assert len(set(new_model.mock.galaxy_table["axis_ratio"])) > 1

        assert "disrupted" in list(new_model.mock.galaxy_table.keys())
        assert set(new_model.mock.galaxy_table["disrupted"]) == {True, False}

        mask = new_model.mock.galaxy_table["disrupted"] == False
        np.testing.assert_allclose(new_model.mock.galaxy_table["axis_ratio"][mask], 0.3)
        assert np.all(new_model.mock.galaxy_table["axis_ratio"] >= 0)
        assert np.all(new_model.mock.galaxy_table["axis_ratio"] <= 1)

        cenmask = new_model.mock.galaxy_table["gal_type"] == "centrals"
        cens = new_model.mock.galaxy_table[cenmask]
        cens_disrupted_mask = cens["disrupted"] == True
        disrupted_cens = cens[cens_disrupted_mask]
        try:
            assert (
                disrupted_cens["halo_mvir"].max()
                <= new_model.param_dict["max_disruption_mass_centrals"]
            )
        except ValueError:
            # in this Monte Carlo realization, there were zero disrupted centrals
            pass

        sats = new_model.mock.galaxy_table[~cenmask]
        sats_disrupted_mask = sats["disrupted"] == True
        disrupted_sats = sats[sats_disrupted_mask]
        try:
            assert (
                disrupted_sats["halo_mvir"].max()
                <= new_model.param_dict["max_disruption_mass_satellites"]
            )
        except ValueError:
            # in this Monte Carlo realization, there were zero disrupted satellites
            pass

    def test_hod_modeling_tutorial5(self):
        class Shape(object):
            def __init__(self, gal_type):

                self.gal_type = gal_type
                self._mock_generation_calling_sequence = ["assign_shape"]
                self._galprop_dtypes_to_allocate = np.dtype([("shape", object)])

            def assign_shape(self, **kwargs):
                table = kwargs["table"]
                with NumpyRNGContext(fixed_seed):
                    randomizer = np.random.random(len(table))
                table["shape"][:] = np.where(randomizer > 0.5, "elliptical", "disk")

        class Size(object):
            def __init__(self, gal_type):

                self.gal_type = gal_type
                self._mock_generation_calling_sequence = ["assign_size"]
                self._galprop_dtypes_to_allocate = np.dtype([("galsize", "f4")])

                self.new_haloprop_func_dict = {
                    "halo_custom_size": self.calculate_halo_size
                }

                # self.list_of_haloprops_needed = ['halo_spin']

            def assign_size(self, **kwargs):
                table = kwargs["table"]
                disk_mask = table["shape"] == "disk"
                table["galsize"][disk_mask] = table["halo_spin"][disk_mask]
                table["galsize"][~disk_mask] = table["halo_custom_size"][~disk_mask]

            def calculate_halo_size(self, **kwargs):
                table = kwargs["table"]
                return 2 * table["halo_rs"]

        from ...empirical_models import Leauthaud11Cens, TrivialPhaseSpace

        cen_occupation = Leauthaud11Cens()
        cen_profile = TrivialPhaseSpace(gal_type="centrals")
        cen_shape = Shape(gal_type="centrals")
        cen_size = Size(gal_type="centrals")

        from ...empirical_models import HodModelFactory

        model = HodModelFactory(
            centrals_occupation=cen_occupation,
            centrals_profile=cen_profile,
            centrals_shape=cen_shape,
            centrals_size=cen_size,
            model_feature_calling_sequence=(
                "centrals_occupation",
                "centrals_profile",
                "centrals_shape",
                "centrals_size",
            ),
        )

        # We forgot to put 'halo_spin' in list_of_haloprops_needed,
        # so attempting to populate a mock should raise an exception
        halocat = FakeSim()
        with pytest.raises(KeyError) as err:
            model.populate_mock(halocat)
        assert "halo_spin" in err.value.args[0]

    def test_hod_modeling_tutorial6(self):
        class Shape(object):
            def __init__(self, gal_type):

                self.gal_type = gal_type
                self._mock_generation_calling_sequence = ["assign_shape"]
                self._galprop_dtypes_to_allocate = np.dtype([("shape", object)])

            def assign_shape(self, **kwargs):
                table = kwargs["table"]
                with NumpyRNGContext(fixed_seed):
                    randomizer = np.random.random(len(table))
                table["shape"][:] = np.where(randomizer > 0.5, "elliptical", "disk")

        class Size(object):
            def __init__(self, gal_type):

                self.gal_type = gal_type
                self._mock_generation_calling_sequence = ["assign_size"]
                self._galprop_dtypes_to_allocate = np.dtype([("galsize", "f4")])

                self.new_haloprop_func_dict = {
                    "halo_custom_size": self.calculate_halo_size
                }

                self.list_of_haloprops_needed = ["halo_spin"]

            def assign_size(self, **kwargs):
                table = kwargs["table"]
                disk_mask = table["shape"] == "disk"
                table["galsize"][disk_mask] = table["halo_spin"][disk_mask]
                table["galsize"][~disk_mask] = table["halo_custom_size"][~disk_mask]

            def calculate_halo_size(self, **kwargs):
                table = kwargs["table"]
                return 2 * table["halo_rs"]

        from ...empirical_models import Leauthaud11Cens, TrivialPhaseSpace

        cen_occupation = Leauthaud11Cens()
        cen_profile = TrivialPhaseSpace(gal_type="centrals")
        cen_shape = Shape(gal_type="centrals")
        cen_size = Size(gal_type="centrals")

        from ...empirical_models import HodModelFactory

        model = HodModelFactory(
            centrals_occupation=cen_occupation,
            centrals_profile=cen_profile,
            centrals_shape=cen_shape,
            centrals_size=cen_size,
            model_feature_calling_sequence=(
                "centrals_occupation",
                "centrals_profile",
                "centrals_shape",
                "centrals_size",
            ),
        )

        halocat = FakeSim()
        model.populate_mock(halocat)
