# -*- coding: utf-8 -*-
""" Module containing container classes for
the simulations provided by Halotools.
The attributes of the classes defined below
are used to attach metadata to the Halotools-provided
halo catalogs as they are loaded into memory.
"""

from abc import ABCMeta
from astropy.extern import six
from astropy import cosmology

__all__ = ('NbodySimulation',
           'Bolshoi', 'BolPlanck', 'MultiDark', 'Consuelo')

supported_sim_list = ('bolshoi', 'bolplanck', 'consuelo', 'multidark')

######################################################
########## Simulation classes defined below ##########
######################################################


@six.add_metaclass(ABCMeta)
class NbodySimulation(object):
    """ Abstract base class for any object used as a container for
    simulation specs.
    """

    def __init__(self, simname, Lbox, particle_mass, num_ptcl_per_dim,
            softening_length, initial_redshift, cosmology):
        """
        Parameters
        -----------
        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

        Lbox : float
            Size of the simulated box in Mpc with h=1 units.

        particle_mass : float
            Mass of the dark matter particles in Msun with h=1 units.

        num_ptcl_per_dim : int
            Number of particles per dimension.

        softening_length : float
            Softening scale of the particle interactions in kpc with h=1 units.

        initial_redshift : float
            Redshift at which the initial conditions were generated.

        cosmology : object
            `astropy.cosmology` instance giving the cosmological parameters
            with which the simulation was run.

        """
        self.simname = simname
        self.Lbox = Lbox
        self.particle_mass = particle_mass
        self.num_ptcl_per_dim = num_ptcl_per_dim
        self.softening_length = softening_length
        self.initial_redshift = initial_redshift
        self.cosmology = cosmology

        self._attrlist = (
            ['simname', 'Lbox', 'particle_mass', 'num_ptcl_per_dim',
            'softening_length', 'initial_redshift', 'cosmology']
            )


class Bolshoi(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5 cosmology
    with Lbox = 250 Mpc/h and particle mass of ~1e8 Msun/h.

    For a detailed description of the
    simulation specs, see http://www.cosmosim.org/cms/simulations/multidark-project/bolshoi.
    """

    def __init__(self):

        super(Bolshoi, self).__init__(simname='bolshoi', Lbox=250.,
            particle_mass=1.35e8, num_ptcl_per_dim=2048,
            softening_length=1., initial_redshift=80., cosmology=cosmology.WMAP5)

        self.orig_ascii_web_location = (
            'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/')


class BolPlanck(NbodySimulation):
    """ Cosmological N-body simulation of Planck 2013 cosmology
    with Lbox = 250 Mpc/h and
    particle mass of ~1e8 Msun/h.

    For a detailed description of the
    simulation specs, see http://www.cosmosim.org/cms/simulations/bolshoip-project/bolshoip/.
    """

    def __init__(self):

        super(BolPlanck, self).__init__(simname='bolplanck', Lbox=250.,
            particle_mass=1.35e8, num_ptcl_per_dim=2048,
            softening_length=1., initial_redshift=80., cosmology=cosmology.Planck13)

        self.orig_ascii_web_location = (
            'http://www.slac.stanford.edu/~behroozi/BPlanck_Hlists/')


class MultiDark(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5 cosmology
    with Lbox = 1Gpc/h and particle mass of ~1e10 Msun/h.

    For a detailed description of the
    simulation specs, see http://www.cosmosim.org/cms/simulations/multidark-project/mdr1.
    """

    def __init__(self):

        super(MultiDark, self).__init__(simname='multidark', Lbox=1000.,
            particle_mass=8.721e9, num_ptcl_per_dim=2048,
            softening_length=7., initial_redshift=65., cosmology=cosmology.WMAP5)

        self.orig_ascii_web_location = (
            'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/')


class Consuelo(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5-like cosmology
    with Lbox = 420 Mpc/h and particle mass of 4e8 Msun/h.

    For a detailed description of the
    simulation specs, see http://lss.phy.vanderbilt.edu/lasdamas/simulations.html.
    """

    def __init__(self):

        super(Consuelo, self).__init__(simname='consuelo', Lbox=420.,
            particle_mass=1.87e9, num_ptcl_per_dim=1400,
            softening_length=8., initial_redshift=99., cosmology=cosmology.WMAP5)

        self.orig_ascii_web_location = (
            'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/')
