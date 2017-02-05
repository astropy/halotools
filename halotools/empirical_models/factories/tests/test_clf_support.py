"""
"""
import numpy as np
from ..prebuilt_model_factory import HodModelFactory
from ...occupation_models.occupation_model_template import OccupationComponent

from ....sim_manager import FakeSim


class DummyCLF(OccupationComponent):
    """ Bare bones class used to verify that HodMockFactory supports CLF-style models.

    This implementation can be used as a baseline pattern to match
    for users writing their own CLF-style models.
    """

    def __init__(self, gal_type, threshold, upper_occupation_bound, prim_haloprop_key):

        super(DummyCLF, self).__init__(gal_type=gal_type, threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key)

        self._mock_generation_calling_sequence = ['mc_occupation', 'mc_luminosity']
        self._galprop_dtypes_to_allocate = np.dtype(
            [('luminosity', 'f4'), ('halo_num_'+gal_type, 'i4')])

    def mean_occupation(self, **kwargs):
        mass = kwargs['table'][self.prim_haloprop_key]
        return np.zeros_like(mass) + 0.05

    def mc_luminosity(self, **kwargs):
        table = kwargs['table']
        table['luminosity'][:] = np.linspace(0, 1, len(table))


def test_clf_support():
    """ Regression test verifying that HodMockFactory supports CLF-style models.

    In particular, the test ensures that sub-classes of OccupationComponent
    can assign additional galaxy properties besides number of galaxies.
    """
    dummy_model_dict = {'centrals_occupation': DummyCLF('centrals', -19, 1., 'halo_mvir')}
    model = HodModelFactory(**dummy_model_dict)

    # Ensure that the additional galaxy property defined by the component model
    # makes it into the _galprop_dtypes_to_allocate of the composite model
    assert 'luminosity' in model._galprop_dtypes_to_allocate.names

    # Ensure that we can populate a mock
    # and that the luminosity assignment occurs as expected
    halocat = FakeSim()
    model.populate_mock(halocat)
    assert np.allclose(model.mock.galaxy_table['luminosity'],
        np.linspace(0, 1, len(model.mock.galaxy_table)))
