r"""
halotools model components for modelling central and scatellite intrinsic alignments
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# vector rotations
from ...utils import rotate_vector_collection
from ...utils.mcrotations import random_perpendicular_directions, random_unit_vectors_3d
from ...utils.vector_utilities import (elementwise_dot, elementwise_norm, normalized_vectors,
                                        angles_between_list_of_vectors)
from ...utils.rotations3d import (vectors_between_list_of_vectors, vectors_normal_to_planes,
                                   rotation_matrices_from_angles)
# watson distribution
from .watson_distribution import DimrothWatson

# utilities
from warnings import warn
from astropy.utils.misc import NumpyRNGContext


__all__ = ('HaloMassCentralAlignmentStrength',
           'RadialSatelliteAlignmentStrength')
__author__ = ('Duncan Campbell',)


class HaloMassCentralAlignmentStrength():
    """
    model for the stregth of alignment for centrals
    """

    def __init__(self, central_alignment_a=0.05, central_alignment_gamma=0.25):
        """
        Parameters
        ==========
        a : float

        alpha : float
        """

        self.gal_type = 'centrals'
        self._mock_generation_calling_sequence = (['assign_central_alignment_strength'])

        self._galprop_dtypes_to_allocate = np.dtype([(str('central_alignment_strength'), 'f4')])

        self.list_of_haloprops_needed = ['halo_mvir']

        self._methods_to_inherit = (['assign_central_alignment_strength'])
        self.param_dict = ({
            'a': central_alignment_a,
            'gamma': central_alignment_gamma})

    def assign_central_alignment_strength(self, **kwargs):
        """
        Parameters
        ==========
        halo_mvir : array_like
            host halo virial mass
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            halo_m = table['halo_mvir']
        else:
            halo_m = kwargs['halo_mvir']

        s = self.alignment_strength_mass_dependence(halo_m)

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['central_alignment_strength'] = 0.0
            table['central_alignment_strength'][mask] = s[mask]
            return table
        else:
            return s

    def alignment_strength_mass_dependence(self, m):
        """
        Parameters
        ==========
        m : array_like
            scaled halo masses

        Returns
        =======
        alignment_strength : numpy.array
            array fo values bounded between [-1,1]
        """

        a = self.param_dict['a']
        gamma = self.param_dict['gamma']
        result = a*np.log10(m)+gamma
        mask = (result < -0.99)
        result[mask]= -0.99
        mask = (result > 0.99)
        result[mask]= 0.99
        return result


class RadialSatelliteAlignmentStrength():
    """
    model for the stregth of alignment of satellites
    """

    def __init__(self,  satellite_alignment_a= 0.8045208899, satellite_alignment_gamma=-0.04322356):
        """
        Parameters
        ==========
        a : float

        alpha : float
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_satellite_alignment_strength'])

        self._galprop_dtypes_to_allocate = np.dtype([(str('satellite_alignment_strength'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_rvir']

        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (['assign_satellite_alignment_strength'])
        self.param_dict = ({
            'a': satellite_alignment_a,
            'gamma': satellite_alignment_gamma})

    def inherit_halocat_properties(self, **kwargs):
        """
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def assign_satellite_alignment_strength(self, **kwargs):
        """
        Parameters
        ==========
        x, y, z : array_like
            galaxy positions

        halo_x, halo_y, halo_z : array_like
            host halo positions

        halo_r : array_like
            host halo virial radius

        Lbox : array_like
            size of simulation along each dimension
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            x = table['x']
            y = table['y']
            z = table['z']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
            halo_r = table['halo_rvir']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        else:
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']
            halo_r = kwargs['halo_rvir']
            Lbox = kwargs['Lbox']

        # define halo-center - satellite vector
        dx = (x - halo_x)
        mask = dx>Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx<-1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (y - halo_y)
        mask = dy>Lbox[1]/2.0
        dy[mask] = dy[mask] - Lbox[1]
        mask = dy<-1.0*Lbox[1]/2.0
        dy[mask] = dy[mask] + Lbox[1]

        dz = (z - halo_z)
        mask = dz>Lbox[2]/2.0
        dz[mask] = dz[mask] - Lbox[2]
        mask = dz<-1.0*Lbox[2]/2.0
        dz[mask] = dz[mask] + Lbox[2]

        # calculate scaled halo virial radius
        r = np.sqrt(dx**2 + dy**2 + dz**2)/halo_r

        s = self.alignment_strength_radial_dependence(r)

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['satellite_alignment_strength'] = 0.0
            table['satellite_alignment_strength'][mask] = s[mask]
            return table
        else:
            return s

    def _alignment_strength_radial_dependence(self, r):
        """
        Parameters
        ==========
        r : array_like
            scaled radial position

        Returns
        =======
        alignment_strength : numpy.array
            array fo values bounded between [-1,1]
        """

        r = np.atleast_1d(r)

        a = self.param_dict['a']
        gamma = self.param_dict['gamma']

        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(r!=0, a*(1.0-1.0/(1.0+(1.0/r)**gamma)), 0.99)

        mask = (result < -0.99)
        result[mask]= -0.99
        mask = (result > 0.99)
        result[mask]= 0.99

        return result

    def alignment_strength_radial_dependence(self, r):
        """
        Parameters
        ==========
        r : array_like
            scaled radial position

        Returns
        =======
        alignment_strength : numpy.array
            array fo values bounded between [-1,1]
        """

        r = np.atleast_1d(r)

        a = self.param_dict['a']
        gamma= self.param_dict['gamma']

        ymax = 0.99
        ymin = -0.99

        result = np.zeros(len(r))
        result = a*(r**gamma)

        mask = (result > ymax)
        result[mask] = ymax

        mask = (result < ymin)
        result[mask] = ymin

        return result


def alignment_strength(p):
    r"""
    convert alignment strength argument to shape parameter for costheta distribution
    """

    p = np.atleast_1d(p)
    k = np.zeros(len(p))
    p = p*np.pi/2.0
    k = np.tan(p)
    mask = (p == 1.0)
    k[mask] = np.inf
    mask = (p == -1.0)
    k[mask] = -1.0*np.inf
    return -1.0*k


def inverse_alignment_strength(k):
    r"""
    convert shape parameter for costheta distribution to alignment strength
    """

    k = np.atleast_1d(k)
    p = np.zeros(len(k))

    k = k
    p = -1.0*np.arctan(k)/(np.pi/2.0)

    return p


