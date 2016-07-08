:orphan:

.. _zheng07_using_cenocc_model_tutorial:

************************************************
Advanced usage of the ``zheng07`` model
************************************************

.. currentmodule:: halotools.empirical_models

As described in the :ref:`zheng07_composite_model` tutorial, the ``zheng07`` model
gives you control over whether the :math:`\langle N_{\rm sat} \rangle` function
is multiplied by a :math:`\langle N_{\rm cen} \rangle` prefactor.
If you are using the `PrebuiltHodModelFactory` with the ``zheng07`` model name,
then your only choice for how :math:`\langle N_{\rm cen} \rangle` prefactor is computed is
the model used in `Zheng et al 2007 <http://arxiv.org/abs/astro-ph/0703457>`_.

However, the ``cenocc_model`` keyword of `Zheng07Sats` allows you
to calculate the :math:`\langle N_{\rm cen} \rangle` prefactor
using *any* `OccupationComponent`, including one of your own design.
To build a composite HOD model using the ``cenocc_model`` keyword,
you must call the `HodModelFactory` directly, rather than the `PrebuiltHodModelFactory`.
In the code below, we will demonstrate an explicit example of how to do so.
Note the similarity of the code below to the `zheng07_model_dictionary` source code.

First we create a new (rather trivial) `OccupationComponent` sub-class that we will
use for our centrals.

.. code::

	from halotools.empirical_models import OccupationComponent
	class MyCenModel(OccupationComponent):

	    def __init__(self, threshold):
	        OccupationComponent.__init__(self, gal_type='centrals',
	        		threshold=threshold, upper_occupation_bound=1.)

	        self.param_dict['new_cen_param'] = 0.5

	    def mean_occupation(self, **kwargs):
	    	halo_table = kwargs['table']
	    	result = np.zeros(len(halo_table)) + self.param_dict['new_cen_param']
	        return result

Now we will build an instance of ``MyCenModel``,
and use the ``cenocc_model`` keyword of `Zheng07Sats` to create a custom version
of this satellite model.

.. code::

	centrals_occupation = MyCenModel(threshold=-20)
	satellites_occupation = Zheng07Sats(threshold=-20, modulate_with_cenocc=True, cenocc_model=centrals_occupation)

Finally, we make the same choices for the profile modeling as made in the normal
pre-built ``zheng07`` composite model, and then pass the resulting collection
of components to the `HodModelFactory`.

.. code::

	from halotools.empirical_models import TrivialPhaseSpace, NFWPhaseSpace
	centrals_profile = TrivialPhaseSpace()
	satellites_profile = NFWPhaseSpace()

	from halotools.empirical_models import HodModelFactory
	model_dict = {'centrals_occupation': centrals_occupation, 'centrals_profile': centrals_profile, 'satellites_occupation': satellites_occupation, 'satellites_profile': satellites_profile}
	composite_model = HodModelFactory(**model_dict)

.. note::

	Halotools provides no checks on the self-consistency between your choice for
	``centrals_occupation`` and the model instance bound to the ``cenocc_model``
	keyword. If you use the ``cenocc_model`` keyword when building a composite HOD model,
	then it is your responsibility to self-consistently use the same model for your
	centrals as for the ``cenocc_model`` argument. If you don't, then changes to the
	values in the ``param_dict`` of the composite model that pertain to
	``centrals_occupation`` will not propagate through to the behavior of ``satellites_occupation``.


