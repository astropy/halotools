"""
"""
import numpy as np

__all__ = ('cumulative_mass_PDF', 'dimensionless_mass_density')


def _g_integral(x):
    r""" Convenience function used to evaluate the profile.

        :math:`g(x) \equiv \rm{ln}(1+x) - x / (1+x)`

    Parameters
    ----------
    x : array_like

    Returns
    -------
    g : array_like

    Examples
    --------
    >>> result = _g_integral(1)
    >>> Npts = 25
    >>> result = _g_integral(np.logspace(-1, 1, Npts))
    """
    x = np.atleast_1d(x).astype(np.float64)
    return np.log(1.0+x) - (x/(1.0+x))


def cumulative_mass_PDF(scaled_radius, conc):
    r"""
    Analytical result for the fraction of the total mass
    enclosed within dimensionless radius of an NFW halo,

    :math:`P_{\rm NFW}(<\tilde{r}) \equiv M_{\Delta}(<\tilde{r}) / M_{\Delta} = g(c\tilde{r})/g(\tilde{r}),`

    where :math:`g(x) \equiv \int_{0}^{x}dy\frac{y}{(1+y)^{2}} = \log(1+x) - x / (1+x)` is computed
    using `g`, and where :math:`\tilde{r} \equiv r / R_{\Delta}`.

    See :ref:`nfw_cumulative_mass_pdf_derivation` for a derivation of this expression.

    Parameters
    -------------
    scaled_radius : array_like
        Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
        :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

    conc : array_like
        Value of the halo concentration. Can either be a scalar, or a numpy array
        of the same dimension as the input ``scaled_radius``.

    Returns
    -------------
    p: array_like
        The fraction of the total mass enclosed
        within the input ``scaled_radius``, in :math:`M_{\odot}/h`;
        has the same dimensions as the input ``scaled_radius``.

    Examples
    --------
    >>> Npts = 100
    >>> scaled_radius = np.logspace(-2, 0, Npts)
    >>> conc = 5
    >>> result = cumulative_mass_PDF(scaled_radius, conc)
    >>> concarr = np.linspace(1, 100, Npts)
    >>> result = cumulative_mass_PDF(scaled_radius, concarr)
    """
    scaled_radius = np.where(scaled_radius > 1, 1, scaled_radius)
    scaled_radius = np.where(scaled_radius < 0, 0, scaled_radius)
    return _g_integral(conc*scaled_radius) / _g_integral(conc)


def dimensionless_mass_density(scaled_radius, conc):
    r"""
    Physical density of the NFW halo scaled by the density threshold of the mass definition.

    The `dimensionless_mass_density` is defined as
    :math:`\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r}) / \rho_{\rm thresh}`,
    where :math:`\tilde{r}\equiv r/R_{\Delta}`.

    For an NFW halo,
    :math:`\tilde{\rho}_{\rm NFW}(\tilde{r}, c) = \frac{c^{3}/3g(c)}{c\tilde{r}(1 + c\tilde{r})^{2}},`

    where :math:`g(x) \equiv \log(1+x) - x / (1+x)` is computed using the `g` function.

    The quantity :math:`\rho_{\rm thresh}` is a function of
    the halo mass definition, cosmology and redshift,
    and is computed via the
    `~halotools.empirical_models.profile_helpers.density_threshold` function.
    The quantity :math:`\rho_{\rm prof}` is the physical mass density of the
    halo profile and is computed via the `mass_density` function.
    See :ref:`nfw_spatial_profile_derivations` for a derivation of this expression.

    Parameters
    -----------
    scaled_radius : array_like
        Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
        :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

    conc : array_like
        Value of the halo concentration. Can either be a scalar, or a numpy array
        of the same dimension as the input ``scaled_radius``.

    Returns
    -------
    dimensionless_density: array_like
        Dimensionless density of a dark matter halo
        at the input ``scaled_radius``, normalized by the
        `~halotools.empirical_models.profile_helpers.density_threshold`
        :math:`\rho_{\rm thresh}` for the
        halo mass definition, cosmology, and redshift.
        Result is an array of the dimension as the input ``scaled_radius``.
    """
    numerator = conc**3/(3.*_g_integral(conc))
    denominator = conc*scaled_radius*(1 + conc*scaled_radius)**2
    return numerator/denominator

