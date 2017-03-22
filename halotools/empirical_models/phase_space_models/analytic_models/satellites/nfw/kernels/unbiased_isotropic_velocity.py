"""
"""
import numpy as np
from scipy.integrate import quad as quad_integration

from .mass_profile import _g_integral


__all__ = ('dimensionless_radial_velocity_dispersion', )


def _jeans_integrand_term1(y):
    r"""
    """
    return np.log(1+y)/(y**3*(1+y)**2)


def _jeans_integrand_term2(y):
    r"""
    """
    return 1/(y**2*(1+y)**3)


def dimensionless_radial_velocity_dispersion(scaled_radius, *conc):
    r"""
    Analytical solution to the isotropic jeans equation for an NFW potential,
    rendered dimensionless via scaling by the virial velocity.

    :math:`\tilde{\sigma}^{2}_{r}(\tilde{r})\equiv\sigma^{2}_{r}(\tilde{r})/V_{\rm vir}^{2} = \frac{c^{2}\tilde{r}(1 + c\tilde{r})^{2}}{g(c)}\int_{c\tilde{r}}^{\infty}{\rm d}y\frac{g(y)}{y^{3}(1 + y)^{2}}`

    See :ref:`nfw_jeans_velocity_profile_derivations` for derivations and implementation details.

    Parameters
    -----------
    scaled_radius : array_like
        Length-Ngals numpy array storing the halo-centric distance
        *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
        :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`.

    conc : float
        Concentration of the halo.

    Returns
    -------
    result : array_like
        Radial velocity dispersion profile scaled by the virial velocity.
        The returned result has the same dimension as the input ``scaled_radius``.
    """
    x = np.atleast_1d(scaled_radius).astype(np.float64)
    result = np.zeros_like(x)

    prefactor = conc*(conc*x)*(1. + conc*x)**2/_g_integral(conc)

    lower_limit = conc*x
    upper_limit = float("inf")
    for i in range(len(x)):
        term1, _ = quad_integration(_jeans_integrand_term1,
            lower_limit[i], upper_limit, epsrel=1e-5)
        term2, _ = quad_integration(_jeans_integrand_term2,
            lower_limit[i], upper_limit, epsrel=1e-5)
        result[i] = term1 - term2

    return np.sqrt(result*prefactor)
