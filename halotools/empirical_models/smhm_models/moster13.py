"""
Module containing classes used to model the mapping between
stellar mass and halo mass based on Moster et al. (2013).
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

from ..component_model_templates import PrimGalpropModel

from ...sim_manager import sim_defaults


__all__ = ['Moster13SmHm']


class Moster13SmHm(PrimGalpropModel):
    """ Stellar-to-halo-mass relation based on
    Moster et al. (2013), arXiv:1205.5807.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing stellar mass.
            Default is set in the `~halotools.empirical_models.model_defaults` module.

        scatter_model : object, optional
            Class governing stochasticity of stellar mass. Default scatter is log-normal,
            implemented by the `LogNormalScatterModel` class.

        scatter_abscissa : array_like, optional
            Array of values giving the abscissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        scatter_ordinates : array_like, optional
            Array of values defining the level of scatter at the input abscissa.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        new_haloprop_func_dict : function object, optional
            Dictionary of function objects used to create additional halo properties
            that may be needed by the model component.
            Used strictly by the `MockFactory` during call to the `process_halo_catalog` method.
            Each dict key of ``new_haloprop_func_dict`` will
            be the name of a new column of the halo catalog; each dict value is a function
            object that returns a length-N numpy array when passed a length-N Astropy table
            via the ``table`` keyword argument.
            The input ``model`` model object has its own new_haloprop_func_dict;
            if the keyword argument ``new_haloprop_func_dict`` passed to `MockFactory`
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to
            ``model``, and exception will be raised.

        """

        super(Moster13SmHm, self).__init__(
            galprop_name='stellar_mass', **kwargs)

        self.publications = ['arXiv:0903.4682', 'arXiv:1205.5807']

    def mean_stellar_mass(self, **kwargs):
        """ Return the stellar mass of a central galaxy as a function
        of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        redshift : float or array, optional
            Redshift of the halo hosting the galaxy.
            Default is set in `~halotools.sim_manager.sim_defaults`.
            If passing an array, must be of the same length as
            the ``prim_haloprop`` or ``table`` argument.

        Returns
        -------
        mstar : array_like
            Array containing stellar masses living in the input table.
        """

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``table`` or ``prim_haloprop``")

        if 'redshift' in list(kwargs.keys()):
            redshift = kwargs['redshift']
        elif hasattr(self, 'redshift'):
            redshift = self.redshift
        else:
            redshift = sim_defaults.default_redshift

        # compute the parameter values that apply to the input redshift
        a = 1./(1+redshift)

        m1 = self.param_dict['m10'] + self.param_dict['m11']*(1-a)
        n = self.param_dict['n10'] + self.param_dict['n11']*(1-a)
        beta = self.param_dict['beta10'] + self.param_dict['beta11']*(1-a)
        gamma = self.param_dict['gamma10'] + self.param_dict['gamma11']*(1-a)

        # Calculate each term contributing to Eqn 2
        norm = 2.*n*mass
        m_by_m1 = mass/(10.**m1)
        denom_term1 = m_by_m1**(-beta)
        denom_term2 = m_by_m1**gamma

        mstar = norm / (denom_term1 + denom_term2)
        return mstar

    def retrieve_default_param_dict(self):
        """ Method returns a dictionary of all model parameters
        set to the values in Table 1 of Moster et al. (2013).

        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """

        d = {
        'm10': 11.590,
        'm11': 1.195,
        'n10': 0.0351,
        'n11': -0.0247,
        'beta10': 1.376,
        'beta11': -0.826,
        'gamma10': 0.608,
        'gamma11': 0.329
        }

        return d
