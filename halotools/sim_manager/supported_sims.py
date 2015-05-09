# -*- coding: utf-8 -*-

import numpy as np
import os, sys, warnings, urllib2, fnmatch

HAS_SOUP = False
try:
    from bs4 import BeautifulSoup
    HAS_SOUP = True
except:
    pass

HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except:
    pass

import posixpath
import urlparse

from . import sim_defaults 

from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import array_like_length as aph_len

from abc import ABCMeta, abstractmethod, abstractproperty
from astropy.extern import six

from astropy import cosmology

__all__ = (
    ['SimulationSpecs', 'Bolshoi', 'BolshoiPl', 'MultiDark', 'Consuelo', 
    'HaloCat', 'BolshoiRockstar', 'BolshoiPlRockstar', 
    'BolshoiBdm', 'MultiDarkRockstar', 'ConsuleoRockstar']
    )


######################################################
########## Simulation classes appear below ########## 
######################################################

@six.add_metaclass(ABCMeta)
class SimulationSpecs(object):
    """ Abstract base class for any object used as a container for 
    simulation specs. 
    """

    def __init__(self, simname):
        self.simname = simname

    @abstractproperty
    def Lbox(self):
        """ Size of the simulated box in Mpc/h. 
        """
        pass

    @abstractproperty
    def particle_mass(self):
        """ Mass of the dark matter particles in Msun/h. 
        """
        pass

    @abstractproperty
    def softening_length(self):
        """ Softening scale of the particle interactions in kpc/h. 
        """
        pass

    @abstractproperty
    def cosmology(self):
        """ Astropy cosmology instance giving the 
        cosmological parameters with which the simulation was run. 
        """
        pass

    @abstractproperty
    def cosmology(self):
        """ Astropy cosmology instance giving the 
        cosmological parameters with which the simulation was run. 
        """
        pass


class Bolshoi(SimulationSpecs):

    def __init__(self):
        super(Bolshoi, self).__init__('bolshoi')

    @property
    def Lbox(self):
        return 250.0

    @property
    def particle_mass(self):
        return 1.35e8

    @property
    def softening_length(self):
        return 1.0

    @property
    def cosmology(self):
        return cosmology.WMAP5

class BolshoiPl(SimulationSpecs):

    def __init__(self):
        super(BolshoiPl, self).__init__('bolshoipl')

    @property
    def Lbox(self):
        return 250.0

    @property
    def particle_mass(self):
        return 1.35e8

    @property
    def softening_length(self):
        return 1.0

    @property
    def cosmology(self):
        return cosmology.Planck13

class MultiDark(SimulationSpecs):

    def __init__(self):
        super(MultiDark, self).__init__('multidark')

    @property
    def Lbox(self):
        return 1000.0

    @property
    def particle_mass(self):
        return 8.7e9

    @property
    def softening_length(self):
        return 7.0

    @property
    def cosmology(self):
        return cosmology.WMAP5

class Consuelo(SimulationSpecs):

    def __init__(self):
        super(Consuelo, self).__init__('consuelo')

    @property
    def Lbox(self):
        return 400.0

    @property
    def particle_mass(self):
        return 4.e8

    @property
    def softening_length(self):
        return 4.0

    @property
    def cosmology(self):
        return cosmology.WMAP5

######################################################
########## Halo-finder classes appear below ########## 
######################################################

@six.add_metaclass(ABCMeta)
class HaloCat(object):

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
        output = fnmatch.filter(catlist, file_pattern)

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

        if aph_len(filename_list)==0:
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
                "This indicates a bug in Halotools, not your usage of the package")
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
                version_name_file_pattern = '*'+version_name+'*'
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

class BolshoiPlRockstar(HaloCat):

    def __init__(self):

        bolshoiPl = BolshoiPl()
        super(BolshoiPlRockstar, self).__init__(bolshoiPl, 'rockstar')

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






























