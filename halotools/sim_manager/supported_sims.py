# -*- coding: utf-8 -*-

import numpy as np
import os, sys, warnings, urllib2, fnmatch

try:
    from bs4 import BeautifulSoup
    HAS_SOUP = True
except ImportError:
    HAS_SOUP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import posixpath
import urlparse

from . import sim_defaults 

from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import array_like_length as custom_len

from abc import ABCMeta, abstractmethod, abstractproperty
from astropy.extern import six

from astropy import cosmology
from astropy import units as u


__all__ = (
    ['NbodySimulation', 'Bolshoi', 'BolshoiPlanck', 'MultiDark', 'Consuelo', 
    'HaloCat', 'BolshoiRockstar', 'BolshoiPlanckRockstar', 
    'BolshoiBdm', 'MultiDarkRockstar', 'ConsuleoRockstar']
    )


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
        simname : string 
            Nickname of the simulation, e.g., 'bolshoi', or 'consuelo'. 

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

    def retrieve_snapshot(self, **kwargs):
        """ Method uses the CatalogManager to return a snapshot object. 
        """
        pass


class Bolshoi(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5 cosmology 
    with Lbox = 250 Mpc/h and particle mass of ~1e8 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://www.cosmosim.org/cms/simulations/multidark-project/bolshoi. 
    """

    def __init__(self):

        super(Bolshoi, self).__init__(simname = 'bolshoi', Lbox = 250., 
            particle_mass = 1.35e8, num_ptcl_per_dim = 2048, 
            softening_length = 1., initial_redshift = 80., cosmology = cosmology.WMAP5)


class BolshoiPlanck(NbodySimulation):
    """ Cosmological N-body simulation of Planck 2013 cosmology 
    with Lbox = 250 Mpc/h and 
    particle mass of ~1e8 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://www.cosmosim.org/cms/simulations/bolshoip-project/bolshoip/. 
    """

    def __init__(self):

        super(BolshoiPlanck, self).__init__(simname = 'bolshoiplanck', Lbox = 250., 
            particle_mass = 1.35e8, num_ptcl_per_dim = 2048, 
            softening_length = 1., initial_redshift = 80., cosmology = cosmology.Planck13)


class MultiDark(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5 cosmology 
    with Lbox = 1Gpc/h and particle mass of ~1e10 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://www.cosmosim.org/cms/simulations/multidark-project/mdr1. 
    """

    def __init__(self):

        super(MultiDark, self).__init__(simname = 'multidark', Lbox = 1000., 
            particle_mass = 8.721e9, num_ptcl_per_dim = 2048, 
            softening_length = 7., initial_redshift = 65., cosmology = cosmology.WMAP5)

class Consuelo(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5-like cosmology 
    with Lbox = 420 Mpc/h and particle mass of 4e8 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://lss.phy.vanderbilt.edu/lasdamas/simulations.html. 
    """

    def __init__(self):

        super(Consuelo, self).__init__(simname = 'consuelo', Lbox = 420., 
            particle_mass = 1.87e9, num_ptcl_per_dim = 1400, 
            softening_length = 8., initial_redshift = 99., cosmology = cosmology.WMAP5)


class SimulationSnapshot(object):

    def __init__(self, simulation_class, redshift):
        """
        """
        simobj = simulation_class()
        for attr in simobj._attrlist:
            setattr(self, attr, getattr(simobj, attr))

        self._attrlist = simobj._attrlist

        self.redshift = redshift
        self._attrlist.append('redshift')


    def retrieve_halocat(self, **kwargs):
        """ Method uses the CatalogManager to return a halo catalog object. 
        """
        pass

    def retrieve_particlecat(self, **kwargs):
        """ Method uses the CatalogManager to return a particle catalog object. 
        """
        pass


######################################################
########## Halo Catalog classes defined below ########## 
######################################################

@six.add_metaclass(ABCMeta)
class HaloCat(object):
    """ Abstract container class for any halo catalog object. 

    Concrete instances of this class are used to standardize the 
    specs of a simulation, how its associated 
    raw ASCII data is read, etc. 
    """

    def __init__(self, simobj, halo_finder):
        self.simulation = simobj
        self.simname = self.simulation.simname

        self.halo_finder = halo_finder

    @abstractproperty
    def original_data_source(self):
        """ String specifying the source of the original, 
        unprocessed halo catalog. For all halo catalogs 
        officially supported by Halotools, this will be a 
        publicly available web location. However, the 
        `~halotools.sim_manager` sub-package provides user-support 
        for proprietary simulations and catalogs. 
        """
        pass

    @property 
    def raw_halocats_available_for_download(self):
        """ Method searches the appropriate web location and 
        returns a list of the filenames of all relevant 
        raw halo catalogs that are available for download. 

        Returns 
        -------
        output : list 
            List of strings of all raw halo catalogs available for download 
            for this simulation and halo-finder. 

        """
        url = self.raw_halocat_web_location

        soup = BeautifulSoup(requests.get(url).text)
        file_list = []
        for a in soup.find_all('a'):
            file_list.append(os.path.join(url, a['href']))

        file_pattern = self.halocat_fname_pattern
        output = fnmatch.filter(file_list, file_pattern)

        return output

    @property 
    def preprocessed_halocats_available_for_download(self):
        """ Method searches the appropriate web location and 
        returns a list of the filenames of all reduced  
        halo catalog binaries processed by Halotools 
        that are available for download. 

        Returns 
        -------
        output : list 
            List of strings of all halo catalogs available for download 
            for this simulation and halo-finder. 

        """
        baseurl = sim_defaults.processed_halocats_webloc
        soup = BeautifulSoup(requests.get(baseurl).text)
        simloclist = []
        for a in soup.find_all('a', href=True):
            dirpath = posixpath.dirname(urlparse.urlparse(a['href']).path)
            if dirpath and dirpath[0] != '/':
                simloclist.append(os.path.join(baseurl, dirpath))

        halocatloclist = []
        for simloc in simloclist:
            soup = BeautifulSoup(requests.get(simloc).text)
            for a in soup.find_all('a', href=True):
                dirpath = posixpath.dirname(urlparse.urlparse(a['href']).path)
                if dirpath and dirpath[0] != '/':
                    halocatloclist.append(os.path.join(simloc, dirpath))

        catlist = []
        for halocatdir in halocatloclist:
            soup = BeautifulSoup(requests.get(halocatdir).text)
            for a in soup.find_all('a'):
                catlist.append(os.path.join(halocatdir, a['href']))

        file_pattern = '*halotools.official.version*'
        all_halocats = fnmatch.filter(catlist, file_pattern)

        file_pattern = '*'+self.simname+'/'+self.halo_finder+'*'
        output = fnmatch.filter(all_halocats, file_pattern)

        return output


    def closest_halocat(self, filename_list, input_redshift, **kwargs):
        """ Method searches `filename_list` and returns the filename 
        of the closest matching snapshot to ``input_redshift``. 

        Parameters 
        ----------
        filename_list : list of strings
            Each entry of the list must be a filenames of 
            the format generated by Rockstar, 
            e.g., `hlist_0.09630.list.gz`. 

        input_redshift : float
            Redshift of the requested snapshot.

        version_name : string, optional
            For cases where multiple versions of the same halo catalog 
            are stored in the input `filename_list`, a matching 
            version name must be supplied to disambiguate. 

        Returns
        -------
        output_fname : list 
            String of the filenames with the closest matching redshift. 

        redshift : float 
            Value of the redshift of the snapshot
        """

        if custom_len(filename_list)==0:
            return None

        # First create a list of floats storing the scale factors of each hlist file
        scale_factor_list = []
        for full_fname in filename_list:
            fname = os.path.basename(full_fname)
            scale_factor_substring = self.get_scale_factor_substring(fname)
            scale_factor = float(scale_factor_substring)
            scale_factor_list.append(scale_factor)
        scale_factor_list = np.array(scale_factor_list)

        # Now use the array utils module to determine 
        # which scale factor is the closest
        input_scale_factor = 1./(1. + input_redshift)
        idx_closest_catalog = find_idx_nearest_val(
            scale_factor_list, input_scale_factor)
        closest_scale_factor = scale_factor_list[idx_closest_catalog]
        output_fname = filename_list[idx_closest_catalog]

        # At this point, we have found a single filename 
        # storing a halo catalog with the most closely matching redshift
        # However, there may be cases where there are multiple versions of a 
        # snapshot in a given location (e.g., the same halo catalog 
        # with different cuts applied). 
        # To robustly handle such cases, 
        # the following behavior will return *all* 
        # filenames in the list which store snapshots 
        # at the most closely matching redshift
        scale_factor_substring = self.get_scale_factor_substring(os.path.basename(output_fname))
        file_pattern = '*'+scale_factor_substring+'*'
        all_matching_fnames = fnmatch.filter(filename_list, file_pattern)
        fnames_with_matching_scale_factor = [os.path.basename(fname) for fname in all_matching_fnames]

        # If necessary, disambiguate by using the input version_name
        if len(fnames_with_matching_scale_factor) == 0:
            raise SyntaxError("No matching filenames found. "
                "This indicates a bug in Halotools, not your usage of the package. "
                "Please raise an Issue on Github, or email a member of the Halotools team.")
        elif len(fnames_with_matching_scale_factor) == 1:
            output_fname = fnames_with_matching_scale_factor[0]
        elif len(fnames_with_matching_scale_factor) > 1:
            if 'version_name' not in kwargs.keys():
                print("\nPrinting all filenames with scale factor = %s:\n" % scale_factor_substring)
                for f in fnames_with_matching_scale_factor:
                    print(f)
                print ("\n")
                raise KeyError("Multiple versions the halo catalog "
                    " were found for scale factor = %s.\n"
                    "In such a case, you must disambiguate by providing"
                    " a string value for keyword argument version_name" 
                    % scale_factor_substring)
            else:
                version_name = kwargs['version_name']
                version_name_file_pattern = '*.list.'+version_name+'.hdf5'
                should_be_unique_fname = fnmatch.filter(
                    fnames_with_matching_scale_factor, version_name_file_pattern)
                if len(should_be_unique_fname) == 0:
                    print("\nPrinting all filenames with scale factor = %s:\n" % scale_factor_substring)
                    for f in fnames_with_matching_scale_factor:
                        print(f)
                    raise KeyError("\nInput version_name = %s.\n"
                        "This does not correspond to any of the version names "
                        "of halo catalogs with scale factor = %s:\n" 
                        % (version_name, scale_factor_substring))
                elif len(should_be_unique_fname)==1:
                    output_fname = should_be_unique_fname[0]
                else:
                    print("\nPrinting all filenames with scale factor = %s:\n" % scale_factor_substring)
                    for f in fnames_with_matching_scale_factor:
                        print(f)
                    raise KeyError("\nInput version_name = %s.\n"
                        "This substring appears in more than one of the "
                        "input fnames" % version_name)

        redshift = (1./closest_scale_factor) - 1
        return os.path.basename(output_fname), redshift

    @abstractproperty
    def halocat_column_info(self):
        """ Method used to define how to interpret the columns of 
        raw ASCII halo catalog data. 

        Returns 
        -------
        dt : numpy dtype
            Numpy dtype object. Each entry is a tuple 
            corresponding to a single column of the ASCII 
            halo catalog. Like all dtype objects, the tuples have 
            just two elements: a field and a data type. 
            The field is a string defining the name of the property 
            stored in the colunmn. The data type can be any type 
            supported by Numpy, e.g., `f4`, `i8`, etc. 

        """
        pass

    @abstractproperty
    def halocat_fname_pattern(self):
        """ String pattern that will be used to identify halo catalog filenames 
        associated with this simulation and halo-finder. 
        """
        pass

    def get_scale_factor_substring(self, fname):
        """ Method extracts the portion of the Rockstar hlist fname 
        that contains the scale factor of the snapshot. 

        Parameters 
        ----------
        fname : string 
            Filename of the hlist. 

        Returns 
        -------
        scale_factor_substring : string 
            The substring specifying the scale factor of the snapshot. 

        Notes 
        -----
        Assumes that the first character of the relevant substring 
        is the one immediately following the first incidence of an underscore, 
        and final character is the one immediately preceding the second decimal. 
        These assumptions are valid for all catalogs currently on the hipacc website, 
        including `bolshoi`, `bolshoi_bdm`, `consuelo`, and `multidark`. 

        """
        first_index = fname.index('_')+1
        last_index = fname.index('.', fname.index('.')+1)
        scale_factor_substring = fname[first_index:last_index]
        return scale_factor_substring

class BolshoiRockstar(HaloCat):
    """ Rockstar-based halo catalog for the Bolshoi simulation. 
    """

    def __init__(self):

        bolshoi = Bolshoi()
        super(BolshoiRockstar, self).__init__(bolshoi, 'rockstar')

    @property 
    def raw_halocat_web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/'

    @property
    def original_data_source(self):
        return self.raw_halocat_web_location

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):

        dt = np.dtype([
            ('scale', 'f4'), 
            ('haloid', 'i8'), 
            ('scale_desc', 'f4'), 
            ('haloid_desc', 'i8'), 
            ('num_prog', 'i4'), 
            ('pid', 'i8'), 
            ('upid', 'i8'), 
            ('pid_desc', 'i8'), 
            ('phantom', 'i4'), 
            ('mvir_sam', 'f4'), 
            ('mvir', 'f4'), 
            ('rvir', 'f4'), 
            ('rs', 'f4'), 
            ('vrms', 'f4'), 
            ('mmp', 'i4'), 
            ('scale_lastmm', 'f4'), 
            ('vmax', 'f4'), 
            ('x', 'f4'), 
            ('y', 'f4'), 
            ('z', 'f4'), 
            ('vx', 'f4'), 
            ('vy', 'f4'), 
            ('vz', 'f4'), 
            ('jx', 'f4'), 
            ('jy', 'f4'), 
            ('jz', 'f4'), 
            ('spin', 'f4'), 
            ('haloid_breadth_first', 'i8'), 
            ('haloid_depth_first', 'i8'), 
            ('haloid_tree_root', 'i8'), 
            ('haloid_orig', 'i8'), 
            ('snap_num', 'i4'), 
            ('haloid_next_coprog_depthfirst', 'i8'), 
            ('haloid_last_prog_depthfirst', 'i8'), 
            ('rs_klypin', 'f4'), 
            ('mvir_all', 'f4'), 
            ('m200b', 'f4'), 
            ('m200c', 'f4'), 
            ('m500c', 'f4'), 
            ('m2500c', 'f4'), 
            ('xoff', 'f4'), 
            ('voff', 'f4'), 
            ('spin_bullock', 'f4'), 
            ('b_to_a', 'f4'), 
            ('c_to_a', 'f4'), 
            ('axisA_x', 'f4'), 
            ('axisA_y', 'f4'), 
            ('axisA_z', 'f4'), 
            ('b_to_a_500c', 'f4'), 
            ('c_to_a_500c', 'f4'), 
            ('axisA_x_500c', 'f4'), 
            ('axisA_y_500c', 'f4'), 
            ('axisA_z_500c', 'f4'), 
            ('t_by_u', 'f4'), 
            ('mass_pe_behroozi', 'f4'), 
            ('mass_pe_diemer', 'f4'), 
            ('macc', 'f4'), 
            ('mpeak', 'f4'), 
            ('vacc', 'f4'), 
            ('vpeak', 'f4'), 
            ('halfmass_scale', 'f4'), 
            ('dmvir_dt_inst', 'f4'), 
            ('dmvir_dt_100myr', 'f4'), 
            ('dmvir_dt_tdyn', 'f4'), 
            ('dmvir_dt_2dtyn', 'f4'), 
            ('dmvir_dt_mpeak', 'f4'), 
            ('scale_mpeak', 'f4'), 
            ('scale_lastacc', 'f4'), 
            ('scale_firstacc', 'f4'), 
            ('mvir_firstacc', 'f4'), 
            ('vmax_firstacc', 'f4'), 
            ('vmax_mpeak', 'f4')
            ])

        return dt

class BolshoiPlanckRockstar(HaloCat):
    """ Rockstar-based halo catalog for the Bolshoi-Planck simulation. 
    """

    def __init__(self):

        bolshoiplanck = BolshoiPlanck()
        super(BolshoiPlanckRockstar, self).__init__(bolshoiplanck, 'rockstar')

    @property 
    def raw_halocat_web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/BPlanck_Hlists/'

    @property
    def original_data_source(self):
        return self.raw_halocat_web_location

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):
        dt = np.dtype([
            ('scale', 'f4'), 
            ('haloid', 'i8'), 
            ('scale_desc', 'f4'), 
            ('haloid_desc', 'i8'), 
            ('num_prog', 'i4'), 
            ('pid', 'i8'), 
            ('upid', 'i8'), 
            ('pid_desc', 'i8'), 
            ('phantom', 'i4'), 
            ('mvir_sam', 'f4'), 
            ('mvir', 'f4'), 
            ('rvir', 'f4'), 
            ('rs', 'f4'), 
            ('vrms', 'f4'), 
            ('mmp', 'i4'), 
            ('scale_lastmm', 'f4'), 
            ('vmax', 'f4'), 
            ('x', 'f4'), 
            ('y', 'f4'), 
            ('z', 'f4'), 
            ('vx', 'f4'), 
            ('vy', 'f4'), 
            ('vz', 'f4'), 
            ('jx', 'f4'), 
            ('jy', 'f4'), 
            ('jz', 'f4'), 
            ('spin', 'f4'), 
            ('haloid_breadth_first', 'i8'), 
            ('haloid_depth_first', 'i8'), 
            ('haloid_tree_root', 'i8'), 
            ('haloid_orig', 'i8'), 
            ('snap_num', 'i4'), 
            ('haloid_next_coprog_depthfirst', 'i8'), 
            ('haloid_last_prog_depthfirst', 'i8'), 
            ('rs_klypin', 'f4'), 
            ('mvir_all', 'f4'), 
            ('m200b', 'f4'), 
            ('m200c', 'f4'), 
            ('m500c', 'f4'), 
            ('m2500c', 'f4'), 
            ('xoff', 'f4'), 
            ('voff', 'f4'), 
            ('spin_bullock', 'f4'), 
            ('b_to_a', 'f4'), 
            ('c_to_a', 'f4'), 
            ('axisA_x', 'f4'), 
            ('axisA_y', 'f4'), 
            ('axisA_z', 'f4'), 
            ('b_to_a_500c', 'f4'), 
            ('c_to_a_500c', 'f4'), 
            ('axisA_x_500c', 'f4'), 
            ('axisA_y_500c', 'f4'), 
            ('axisA_z_500c', 'f4'), 
            ('t_by_u', 'f4'), 
            ('mass_pe_behroozi', 'f4'), 
            ('mass_pe_diemer', 'f4'), 
            ('macc', 'f4'), 
            ('mpeak', 'f4'), 
            ('vacc', 'f4'), 
            ('vpeak', 'f4'), 
            ('halfmass_scale', 'f4'), 
            ('dmvir_dt_inst', 'f4'), 
            ('dmvir_dt_100myr', 'f4'), 
            ('dmvir_dt_tdyn', 'f4'), 
            ('dmvir_dt_2dtyn', 'f4'), 
            ('dmvir_dt_mpeak', 'f4'), 
            ('scale_mpeak', 'f4'), 
            ('scale_lastacc', 'f4'), 
            ('scale_firstacc', 'f4'), 
            ('mvir_firstacc', 'f4'), 
            ('vmax_firstacc', 'f4'), 
            ('vmax_mpeak', 'f4')
            ])
        return dt

class BolshoiBdm(HaloCat):
    """ BDM-based halo catalog for the Bolshoi simulation. 
    """

    def __init__(self):

        bolshoi = Bolshoi()
        super(BolshoiBdm, self).__init__(bolshoi, 'bdm')

    @property 
    def raw_halocat_web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/'

    @property
    def original_data_source(self):
        return self.raw_halocat_web_location

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):
        dt = np.dtype([
            ('scale', 'f4'), 
            ('haloid', 'i8'), 
            ('scale_desc', 'f4'), 
            ('haloid_desc', 'i8'), 
            ('num_prog', 'i4'), 
            ('pid', 'i8'), 
            ('upid', 'i8'), 
            ('pid_desc', 'i8'), 
            ('phantom', 'i4'),  
            ('mvir_sam', 'f4'), 
            ('mvir', 'f4'), 
            ('rvir', 'f4'),  
            ('rs', 'f4'), 
            ('vrms', 'f4'), 
            ('mmp', 'i4'), 
            ('scale_lastmm', 'f4'), 
            ('vmax', 'f4'), 
            ('x', 'f4'), 
            ('y', 'f4'), 
            ('z', 'f4'), 
            ('vx', 'f4'), 
            ('vy', 'f4'), 
            ('vz', 'f4'), 
            ('jx', 'f4'), 
            ('jy', 'f4'), 
            ('jz', 'f4'), 
            ('spin', 'f4'), 
            ('haloid_breadth_first', 'i8'), 
            ('haloid_depth_first', 'i8'), 
            ('haloid_tree_root', 'i8'), 
            ('haloid_orig', 'i8'), 
            ('snap_num', 'i4'), 
            ('haloid_next_coprog_depthfirst', 'i8'), 
            ('haloid_last_prog_depthfirst', 'i8'), 
            ('xoff', 'f4'), 
            ('2K/Ep-1', 'f4'), 
            ('Rrms', 'f4'), 
            ('b_to_a', 'f4'), 
            ('c_to_a', 'f4'), 
            ('axisA_x', 'f4'), 
            ('axisA_y', 'f4'), 
            ('axisA_z', 'f4'), 
            ('macc', 'f4'), 
            ('mpeak', 'f4'), 
            ('vacc', 'f4'), 
            ('vpeak', 'f4')
            ]) 
        return dt

class MultiDarkRockstar(HaloCat):
    """ Rockstar-based halo catalog for the Multidark simulation. 
    """

    def __init__(self):

        multidark = MultiDark()
        super(MultiDarkRockstar, self).__init__(multidark, 'rockstar')

    @property 
    def raw_halocat_web_location(self):
        return 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/'

    @property
    def original_data_source(self):
        return self.raw_halocat_web_location

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):
        dt = np.dtype([
            ('scale', 'f4'), 
            ('haloid', 'i8'), 
            ('scale_desc', 'f4'), 
            ('haloid_desc', 'i8'), 
            ('num_prog', 'i4'), 
            ('pid', 'i8'), 
            ('upid', 'i8'), 
            ('pid_desc', 'i8'), 
            ('phantom', 'i4'), 
            ('mvir_sam', 'f4'), 
            ('mvir', 'f4'), 
            ('rvir', 'f4'), 
            ('rs', 'f4'), 
            ('vrms', 'f4'), 
            ('mmp', 'i4'), 
            ('scale_lastmm', 'f4'), 
            ('vmax', 'f4'), 
            ('x', 'f4'), 
            ('y', 'f4'), 
            ('z', 'f4'), 
            ('vx', 'f4'), 
            ('vy', 'f4'), 
            ('vz', 'f4'), 
            ('jx', 'f4'), 
            ('jy', 'f4'), 
            ('jz', 'f4'), 
            ('spin', 'f4'), 
            ('haloid_breadth_first', 'i8'), 
            ('haloid_depth_first', 'i8'), 
            ('haloid_tree_root', 'i8'), 
            ('haloid_orig', 'i8'), 
            ('snap_num', 'i4'), 
            ('haloid_next_coprog_depthfirst', 'i8'), 
            ('haloid_last_prog_depthfirst', 'i8'), 
            ('rs_klypin', 'f4'), 
            ('mvir_all', 'f4'), 
            ('m200b', 'f4'), 
            ('m200c', 'f4'), 
            ('m500c', 'f4'), 
            ('m2500c', 'f4'), 
            ('xoff', 'f4'), 
            ('voff', 'f4'), 
            ('spin_bullock', 'f4'), 
            ('b_to_a', 'f4'), 
            ('c_to_a', 'f4'), 
            ('axisA_x', 'f4'), 
            ('axisA_y', 'f4'), 
            ('axisA_z', 'f4'), 
            ('b_to_a_500c', 'f4'), 
            ('c_to_a_500c', 'f4'), 
            ('axisA_x_500c', 'f4'), 
            ('axisA_y_500c', 'f4'), 
            ('axisA_z_500c', 'f4'), 
            ('t_by_u', 'f4'), 
            ('mass_pe_behroozi', 'f4'), 
            ('mass_pe_diemer', 'f4'), 
            ('macc', 'f4'), 
            ('mpeak', 'f4'), 
            ('vacc', 'f4'), 
            ('vpeak', 'f4'), 
            ('halfmass_scale', 'f4'), 
            ('dmvir_dt_inst', 'f4'), 
            ('dmvir_dt_100myr', 'f4'), 
            ('dmvir_dt_tdyn', 'f4'), 
            ('dmvir_dt_2dtyn', 'f4'), 
            ('dmvir_dt_mpeak', 'f4'), 
            ('scale_mpeak', 'f4'), 
            ('scale_lastacc', 'f4'), 
            ('scale_firstacc', 'f4'), 
            ('mvir_firstacc', 'f4'), 
            ('vmax_firstacc', 'f4'), 
            ('vmax_mpeak', 'f4')
            ])
        return dt

class ConsuleoRockstar(HaloCat):
    """ Rockstar-based halo catalog for the Consuelo simulation. 
    """

    def __init__(self):

        consuelo = Consuelo()
        super(ConsuleoRockstar, self).__init__(consuelo, 'rockstar')

    @property 
    def raw_halocat_web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'

    @property
    def original_data_source(self):
        return self.raw_halocat_web_location

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):
        dt = np.dtype([
            ('scale', 'f4'), 
            ('haloid', 'i8'), 
            ('scale_desc', 'f4'), 
            ('haloid_desc', 'i8'), 
            ('num_prog', 'i4'), 
            ('pid', 'i8'), 
            ('upid', 'i8'), 
            ('pid_desc', 'i8'), 
            ('phantom', 'i4'),  
            ('mvir_sam', 'f4'), 
            ('mvir', 'f4'), 
            ('rvir', 'f4'),  
            ('rs', 'f4'), 
            ('vrms', 'f4'), 
            ('mmp', 'i4'), 
            ('scale_lastmm', 'f4'), 
            ('vmax', 'f4'), 
            ('x', 'f4'), 
            ('y', 'f4'), 
            ('z', 'f4'), 
            ('vx', 'f4'), 
            ('vy', 'f4'), 
            ('vz', 'f4'), 
            ('jx', 'f4'), 
            ('jy', 'f4'), 
            ('jz', 'f4'), 
            ('spin', 'f4'), 
            ('haloid_breadth_first', 'i8'), 
            ('haloid_depth_first', 'i8'), 
            ('haloid_tree_root', 'i8'), 
            ('haloid_orig', 'i8'), 
            ('snap_num', 'i4'), 
            ('haloid_next_coprog_depthfirst', 'i8'), 
            ('haloid_last_prog_depthfirst', 'i8'), 
            ('rs_klypin', 'f4'), 
            ('mvir_all', 'f4'), 
            ('m200b', 'f4'), 
            ('m200c', 'f4'), 
            ('m500c', 'f4'), 
            ('m2500c', 'f4'), 
            ('xoff', 'f4'), 
            ('voff', 'f4'), 
            ('spin_bullock', 'f4'), 
            ('b_to_a', 'f4'), 
            ('c_to_a', 'f4'), 
            ('axisA_x', 'f4'), 
            ('axisA_y', 'f4'), 
            ('axisA_z', 'f4'), 
            ('t_by_u', 'f4'), 
            ('macc', 'f4'), 
            ('mpeak', 'f4'), 
            ('vacc', 'f4'), 
            ('vpeak', 'f4'), 
            ('scale_halfmass', 'f4')
            ])
        return dt






























