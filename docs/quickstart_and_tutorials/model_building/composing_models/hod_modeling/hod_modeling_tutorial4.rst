.. _hod_modeling_tutorial4:

****************************************************************
Example 4: A more complex HOD component model 
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the :ref:`hod_modeling_tutorial0`, 
illustrates a more complex example of a component model that 
that you have written yourself. What follows is basically just 
a more full-features version of the previous 
:ref:`hod_modeling_tutorial3` that illustrates a few more tricks. 

.. code:: python

    class Shape(object):
        
        def __init__(self, gal_type, prim_haloprop_key):

            self.gal_type = gal_type
            self._mock_generation_calling_sequence = (
                ['assign_disrupted', 'assign_axis_ratio'])
            self._galprop_dtypes_to_allocate = np.dtype(
                [('axis_ratio', 'f4'), ('disrupted', bool)])
            self.list_of_haloprops_needed = ['halo_spin']

            self.prim_haloprop_key = prim_haloprop_key
            self._methods_to_inherit = (
                ['assign_disrupted', 'assign_axis_ratio', 
                'disrupted_fraction_vs_halo_mass'])
            self.param_dict = ({
                'max_disruption_mass_'+self.gal_type: 1e12, 
                'disrupted_fraction_'+self.gal_type: 0.25})

        def assign_disrupted(self, **kwargs):
            if 'table' in kwargs.keys():
                table = kwargs['table']
                halo_mass = table[self.prim_haloprop_key]
            else:
                halo_mass = kwargs['prim_haloprop']

            disrupted_fraction = self.disrupted_fraction_vs_halo_mass(halo_mass)
            randomizer = np.random.uniform(0, 1, len(halo_mass))
            is_disrupted = randomizer < disrupted_fraction

            if 'table' in kwargs.keys():
                table['disrupted'][:] = is_disrupted
            else:
                return is_disrupted

        def assign_axis_ratio(self, **kwargs):
            table = kwargs['table']
            mask = table['disrupted'] == True
            num_disrupted = len(table['disrupted'][mask])
            table['axis_ratio'][mask][:] = np.random.random(num_disrupted)
            table['axis_ratio'][~mask][:] = 0.3

        def disrupted_fraction_vs_halo_mass(self, mass):
            bool_mask = mass > self.param_dict['max_disruption_mass_'+self.gal_type]
            val = self.param_dict['disrupted_fraction_'+self.gal_type]
            return np.where(bool_mask == True, 0, val)







            
