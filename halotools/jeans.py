#!/usr/bin/env python

"""

Module encoding jeans analysis for spherically symmetric halos with spherically symmetric
distributions of tracer particles.

Author: Surhud More (surhudkicp@gmail.com), Kavli IPMU

Bugs: Report to the above email address

"""

__all__ = ['dsigmasq']

import scipy.integrate as si
import scipy.interpolate as sint
from scipy import interpolate
import numpy as np
from math import *
from pylab import *

# Some physical constants
gee=4.2994E-9

def dsigmasq(rrp,nsat_spl,massprof_spl,norm):
    """ Function returns LHS of the following integration:

    :math:`\\sigma^2(r|M) = \\frac{4\\pi G}{N_{sat}(r | M)}\\int_{r}^{\\infty}\\frac{dr'}{r'^{2}}N_{sat}(r' | M) M(<r')`

    Parameters 
    ----------
    rrp : array_like 
        An array with radii at which dsigma^s(r|M) needs to be computed

    nsat_spl: array_like
        Spline with the number density distribution of satellites

    massprof_spl: array_like
        Spline with the mass profile M(<r)

    Returns 
    -------
    result : array_like
        d[Sigma^2(r|M)]

    """
    nsat_part=nsat_spl(rrp)
    mass_part=massprof_spl(rrp)
    return norm*nsat_part*4*np.pi*gee*mass_part/rrp^2

"""

Implemented integration equation:
sigma^2(r|M) = 1/nsat(r|M) \int_r^{\infty} nsat(r'|M) 4 pi G M(<r')/r'^2 dr'

:Parameters:

-    rr: An array with radii at which the the number density distribution of
     satellites and the mass profile (M[<rr]) are calculated
-    nsat: Number density of satellites at a given radius rr
-    massprof: Mass within a given radius rr
-    rr_compute: Radii at which sigma^2 should be computed

:Returns:

-    res_arr: Sigma^2(rr_compute|nsat,massprof) in (km/s)^2

:Examples:

"""
def sigmasq(rr, nsat, massprof, rr_compute):

    # Do not raise a bouunds error, but just assume the value to be zero when
    # out of bounds
    nsat_spl=sint.interp1d(rr,nsat,bounds_error=0,fill_value=0,kind='cubic')
    massprof_spl=sint.interp1d(rr,massprof,bounds_error=0,fill_value=0,kind='cubic')

    res_arr=rr_compute*0.0

    for i in range(rr_compute.size):
        norm=1./nsat_spl(rr_compute[i])
        res,err=si.quad(dsigmasq,rr_compute[i],rr[-1],epsrel=1E-3,args=(nsat_spl,massprof_spl,norm))
        res_arr[i]=res

    return res_arr

"""

Implemented integration equation:
sigma_los^2(rap|M) = N/D
D = \int_rap^{Rvir} nsat(r'|M) 2r'/(r'^2-rap^2)^{1/2} dr'

Returns dnsat at a distance rr,rr+drr from the center and where the line of
sight distance is rap

:Parameters:

-    rr: An array with radii at which dnsat needs to be calculated
-    rap: The fixed projected radius at which dnsat is calculated
-    nsat_spl: Spline with the number density distribution of satellites

"""
def dnsat(rr,rap,nsat_spl):
    return nsat_spl(rr)*2*rr/(rr**2-rap**2)**0.5

"""

Implemented integration equation:
sigma_los^2(rap|M) = N/D
N = \int_rap^{Rvir} nsat(r'|M) sigmasq(r|M) 2r'/(r'^2-rap^2)^{1/2} dr'

Returns dN, which needs to be integrated in order to obtain the numerator of the
line of sight velocity dispersion

:Parameters:

-    rr: An array with radii at which the the number density distribution of
     satellites and the mass profile (M[<rr]) are calculated
-    rap: The fixed projected radius at which dnsat is calculated
-    nsat_spl: Spline with the number density distribution of satellites
-    sigmasq_spl: Spline with the sigma^2(r) is initialized

"""
def dsigmasq_los(rr,rap,nsat_spl,sigmasq_spl):
    return sigmasq_spl(rr)*nsat_spl(rr)*2*rr/(rr**2-rap**2)**0.5

"""

Implemented integration equation:
sigma_los^2(rap|M) = N/D
N = \int_rap^{Rvir} nsat(r'|M) sigmasq(r|M) 2r'/(r'^2-rap^2)^{1/2} dr'
D = \int_rap^{Rvir} nsat(r'|M) 2r'/(r'^2-rap^2)^{1/2} dr'

The integral limit can be changed to an appropriate number other than Rvir if
desired

:Parameters:

-    rr: An array with radii at which the the number density distribution of
     satellites and the mass profile (M[<rr]) are calculated, you can provide rr
     in units of Rvir, too
-    nsat: Number density of satellites at a given radius rr
-    massprof: Mass within a given radius rr
-    rap_compute: Radii at which sigma^2_los should be computed
-    los_integral_limit: Provide the line-of-sight integral limit, default is
     the maximum of rr. If rr is in units of Rvir and you want to integrate to
     Rvir, then provide los_integral_limit=1

:Returns:

-    res_arr: Sigma^2_los(rap_compute|nsat,massprof) in (km/s)^2

:Examples:

"""
def sigmasq_los(rr, nsat, massprof, rap_compute, los_integral_limit=np.inf):

    if(los_integral_limit==np.inf):
        los_integral_limit=rr[-1]

    # First obtain the spline for nsat(r)
    nsat_spl=sint.interp1d(rr,nsat,bounds_error=0,fill_value=0,kind='cubic')

    # Then obtain the spline for sigma^2(r)
    sigmasq_r=sigmasq(rr,nsat,massprof,rr)
    sigmasq_spl=sint.interp1d(rr,sigmasq_r,bounds_error=0,fill_value=0,kind='cubic')

    sigmasq_los_res=rap_compute*0.0
    for i in range(rr_compute.size):
        denominator,err=si.quad(dnsat,rap_compute[i],los_integral_limit,epsrel=1E-3,args=(rap_compute[i],nsat_spl))
        numerator,err=si.quad(dsigmasq_los,rap_compute[i],los_integral_limit,epsrel=1E-3,args=(rap_compute[i],nsat_spl,sigmasq_spl))
        sigmasq_los_res[i]=numerator/denominator

    return sigmasq_los_res
