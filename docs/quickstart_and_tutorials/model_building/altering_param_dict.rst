:orphan:

.. _altering_param_dict:

*************************************************
Changing Composite Model Parameters
*************************************************

All Halotools composite models have a ``param_dict`` attribute, 
a python dictionary storing the complete collection 
of tunable parameters of the model. By altering the values of stored 
in ``param_dict``, you change the behavior of the model. 
If you are running an MCMC-type likelihood analysis, 
your walker should explore the parameter space of (some subset of) 
the parameters stored in the model ``param_dict``. 

For information about how changes in ``param_dict`` propagate through 
Halotools source code, see :ref:`param_dict_mechanism` section 
of the :ref:`composite_model_constructor_bookkeeping_mechanisms` page 
in the documentation. 




