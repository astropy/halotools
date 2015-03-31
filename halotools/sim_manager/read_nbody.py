# -*- coding: utf-8 -*-
"""
Methods and classes to load halo and particle catalogs into memory.

"""

__all__=['processed_snapshot','Catalog_Manager']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

from astropy.io import fits as fits
from astropy.table import Table
from astropy.utils.data import get_readable_fileobj
from astropy.utils.data import _get_download_cache_locs as get_download_cache_locs
from astropy.utils.data import _open_shelve as open_shelve

import numpy as np

import configuration
import os, sys, warnings, urllib2
import sim_defaults


class processed_snapshot(object):
    """ Class containing halo and particle data taken from 
    a single snapshot of some Nbody simulation.
    """

    def __init__(self,
        simname=sim_defaults.default_simulation_name,
        scale_factor=sim_defaults.default_scale_factor,
        halo_finder=sim_defaults.default_halo_finder,
        download_yn=True, **kwargs):

        self.simulation_name = simname
        self.scale_factor = scale_factor
        self.halo_finder = halo_finder

        self.download_yn = download_yn

        catman = Catalog_Manager()
        self.catalog_manager = catman

        Lbox, mp, softening = catman.get_simulation_properties(self.simulation_name)
        self.Lbox = Lbox
        self.particle_mass = mp
        self.softening_length = softening

        halo_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache('subhalos',
                scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )

        # If there are no matching halo catalogs in cache,
        # set the halo catalog to the default halo catalog
        if (halo_catalog_filename==None) or (closest_scale_factor != self.scale_factor):
            halo_catalog_filename = catman.default_halo_catalog_filename
            # Download the catalog, if desired
            if download_yn==True:
                catman.download_all_default_catalogs()

        self.halo_catalog_filename = halo_catalog_filename
        self.halo_catalog_dirname = configuration.get_catalogs_dir('subhalos')

        particle_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache('particles',
                scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )

        # If there are no matching particle catalogs in cache,
        # set the particle catalog to the default particle catalog
        if (particle_catalog_filename==None) or (closest_scale_factor != self.scale_factor):
            particle_catalog_filename = catman.default_particle_catalog_filename
            # Download the catalog, if desired
            if download_yn==True:
                catman.download_all_default_catalogs()

        self.particle_catalog_filename = particle_catalog_filename
        self.particle_catalog_dirname = configuration.get_catalogs_dir('particles')

    @property
    def particles(self):
        """ Method to load simulation particle data into memory. 

        The property decorator syntax allows this method to be called 
        as if it is an attribute.

        """

        catalog_type = 'particles'
        particles = self.catalog_manager.load_catalog(catalog_type,
            dirname = self.particle_catalog_dirname,
            filename=self.particle_catalog_filename,
            download_yn = self.download_yn)

        return particles


    @property
    def halos(self):
        """ Method to load simulation halo catalog into memory. 

        The property decorator syntax allows this method to be called 
        as if it is an attribute.

        """
        catalog_type = 'subhalos'
        halos = self.catalog_manager.load_catalog(catalog_type,
            dirname = self.halo_catalog_dirname,
            filename=self.halo_catalog_filename,
            download_yn = self.download_yn)

        return halos

###################################################################################################

class Catalog_Manager(object):
    """ Container class for managing I/O of halo & particle catalogs.
    """

    def __init__(self):
        self.slac_urls = {'bolshoi_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/',
        'bolshoi_bdm_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/',
        'multidark_halos' : 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/',
        'consuelo_halos' : 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'
        }

        self.default_halo_catalog_filename = (
            sim_defaults.default_simulation_name+'_a'+
            str(np.round(sim_defaults.default_scale_factor,4))+'_'+
            sim_defaults.default_halo_finder+'_subhalos.hdf5')

        self.default_particle_catalog_filename = (
            sim_defaults.default_simulation_name+'_a'+
            str(np.round(sim_defaults.default_scale_factor,4))+
            '_particles.fits')


    def retrieve_catalog_filenames_from_url(self,url,catalog_type='subhalos'):
        """ Get the full list of filenames available at the provided url.

        This method uses BeautifulSoup to query the provided url for the list of files stored there. 
        Filenames of halo catalogs are expected to begin with 'hlist', or they will be ignored; 
        filenames of merger trees are expected to begin with 'tree'.

        Parameters 
        ----------
        url : string 
            Web address pointing to the directory to be searched.

        catalog_type : string 
            Specifies what type of catalog to look for at the provided url.

        Returns 
        -------
        file_list : list 
            List of strings of filenames available for downloaded at the provided url.
        """

        from bs4 import BeautifulSoup
        import requests

        soup = BeautifulSoup(requests.get(url).text)
        file_list = []

        ##################
        ### SLAC url case
        if url==sim_defaults.behroozi_web_location:
            ### Set naming conventions of the files hosted at SLAC
            if (catalog_type == 'subhalo') or (catalog_type=='subhalos'): 
                expected_filename_prefix = 'hlist_'
            elif (catalog_type == 'tree') or (catalog_type == 'trees'):
                expected_filename_prefix = 'tree_'
            else:
                raise TypeError("Input catalog type must either be 'subhalos' or 'trees'")
            ### Grab all filenames with the assumed prefix
            for a in soup.find_all('a'):
                link = a['href']
                if link[0:len(expected_filename_prefix)]==expected_filename_prefix: 
                    file_list.append(a['href'])
        ##################
        ### APH url case (simpler, since only two default catalogs are hosted here)
        elif url==sim_defaults.aph_web_location:
            ### Set naming conventions of the files hosted at Yale
            if (catalog_type == 'subhalo') or (catalog_type=='subhalos'): 
                expected_filename_suffix = 'halos.fits'
            elif (catalog_type == 'particle') or (catalog_type=='particles'):
                expected_filename_suffix = 'particles.fits'
            else:
                expected_filename_suffix = '.fits'
            ### Grab all filenames with the assumed suffix
            for a in soup.find_all('a'):
                link = a['href']
                if link[-len(expected_filename_suffix):]==expected_filename_suffix: 
                    file_list.append(a['href'])
        ##################
        ### Some other url managed by the user
        else:
            for a in soup.find_all('a'):
                link = a['href']
                file_list.append(a['href'])

        return file_list
        ##################

    def get_simulation_properties(self,simname):
        """ Return a few characteristics of the input simulation.

        Parameters 
        ----------
        simname : string 
            Specifies the simulation of interest, e.g., 'bolshoi'.

        Returns 
        -------
        Lbox : float 
            Box size in Mpc/h.

        particle_mass : float 
            Particle mass in Msun/h.

        softening : float 
            Softening length in kpc/h.

        """

        Lbox, particle_mass, softening = None, None, None

        if (simname=='bolshoi'):
            Lbox = 250.0
            particle_mass = 1.35e8
            softening = 1.0

        return Lbox, particle_mass, softening 

    def id_rel_cats(self,catalog_type=None,
        simname=None,halo_finder=None):

        # Fix possible pluralization mistake of user
        if catalog_type == 'subhalo': catalog_type='subhalos'
        if catalog_type == 'particle': catalog_type='particles'

        # Identify all catalogs currently stored in the cache directory
        available_catalogs = np.array(
            configuration.list_of_catalogs_in_cache(catalog_type=catalog_type))

        ######### 
        # Define a simple string manipulating function that will be used to 
        # parse a filename into the features defining the catalog
        def parse_fname(s,extname):
            if extname[0] != '.':
                extname = '.'+extname
            s = s[:-len(extname)]

            segment_separations = []
            # Loop over the string to identify the segments
            for idx, char in enumerate(s):
                if char=='_':
                    segment_separations.append(idx)

            segments = []
            first_idx=0
            for idx in segment_separations:
                last_idx=idx
                segment = s[slice(first_idx,last_idx)]
                segments.append(segment)
                first_idx=last_idx+1
            last_idx=None
            segment = s[slice(first_idx,last_idx)]
            segments.append(segment)

            return segments

        #########
        # The file_mask array will determine which of the available catalogs 
        # pass the input specifications
        file_mask  = np.ones(len(available_catalogs),dtype=bool)

        extension = '.hdf5'
        for catidx,catname in enumerate(available_catalogs):
            parsed_string = parse_fname(catname,extension)
            ###
            sim_string = parsed_string[0]
            if (simname != None) & (sim_string != simname):
                file_mask[catidx] = False
            ###
            halo_finder_string = parsed_string[2]
            if (halo_finder != None) & (halo_finder != halo_finder_string) :
                file_mask[catidx] = False

        relevant_catalogs = available_catalogs[file_mask]
        return relevant_catalogs



    def identify_relevant_catalogs(self,catalog_type=None,
        simname=None,halo_finder=None):
        """ Look in cache for any catalog that matches the inputs.

        Parameters 
        ----------
        catalog_type : string 
            Specifies whether we are interested in halo or particle catalogs. 

        simname : string 
            Specifies the simulation of interest, e.g., 'bolshoi'.

        halo_finder : string 
            Specifies the halo-finder used to generate the catalog. 

        Returns 
        -------
        relevant_catalogs : array
            array of strings of the filenames of catalogs matching the input specifications.
        """

        # Fix possible pluralization mistake of user
        if catalog_type == 'subhalo': catalog_type='subhalos'
        if catalog_type == 'particle': catalog_type='particles'

        # Identify all catalogs currently stored in the cache directory
        available_catalogs = np.array(
            configuration.list_of_catalogs_in_cache(catalog_type=catalog_type))

        #########
        # The file_mask array will determine which of the available catalogs 
        # pass the input specifications
        file_mask  = np.ones(len(available_catalogs),dtype=bool)

        # Impose halos vs. particles restriction
        extension = '.hdf5'
        if catalog_type != None:
            last_characters_of_filename=catalog_type+extension
            for ii,c in enumerate(available_catalogs):
                if c[-len(catalog_type)-len(extension):] != last_characters_of_filename:
                    file_mask[ii]=False

       # Impose simulation name restriction
        if simname != None:
            first_characters_of_filename = simname
            for ii,c in enumerate(available_catalogs):
                if (c[0:len(simname)] != first_characters_of_filename) or (c[len(simname)] != '_'):
                    file_mask[ii]=False

        # Impose halo finder restriction
        if halo_finder != None:
            for ii,c in enumerate(available_catalogs):
                if c[-11-len(halo_finder):-11] != halo_finder:
                    file_mask[ii]=False
        relevant_catalogs = available_catalogs[file_mask]

        return relevant_catalogs


    def find_nearest_snapshot_in_cache(self,catalog_type,
        scale_factor=None,redshift=None,
        simname=sim_defaults.default_simulation_name,
        halo_finder = sim_defaults.default_halo_finder):

        """ Identify the catalog in the cache directory with the 
        closest redshift to the requested redshift.

        Returns 
        ------- 
        filename : string 
            filename of pre-processed catalog in cache directory with closest redshift to 
            the requested redshift

        nearest_snapshot : float
            Value of the scale factor of the returned catalog

        """

        # Fix possible pluralization mistake of user
        if catalog_type == 'subhalo': catalog_type='subhalos'
        if catalog_type == 'particle': catalog_type='particles'

        if (scale_factor == None):
            if (redshift == None):
                raise IOError("Must specify either a redshift or a scale factor")
            else:
                scale_factor = 1./(1.+redshift)
        else:
            if (redshift != None):
                raise IOError("Cannot specify both a redshift and a scale factor")

        # Ignore and over-write the halo_finder if looking for particle data
        # This is necessary or else the relevant_catalog finder will bail
        if catalog_type=='particles':
            halo_finder=None

        relevant_catalogs = self.id_rel_cats(
            catalog_type=catalog_type,simname=simname,halo_finder=halo_finder)

        if len(relevant_catalogs)==0:
            if catalog_type=='subhalos':
                warnings.warn("Zero halo catalogs in cache match the input simname & halo-finder")
                return None, None
            elif catalog_type=='particles':
                warnings.warn("Zero particle catalogs in cache match the input simname")
                return None, None

        first_scale_factor_index=len(simname)+2
        last_scale_factor_index = first_scale_factor_index + 6
        available_snapshots = (
            [float(a[first_scale_factor_index:last_scale_factor_index]) 
            for a in relevant_catalogs] )

        idx_nearest_snapshot = np.abs(np.array(available_snapshots)-scale_factor).argmin()
        nearest_snapshot = available_snapshots[idx_nearest_snapshot]
        filename_of_nearest_snapshot = relevant_catalogs[idx_nearest_snapshot]


        # Warn the user if the nearest scale factor differs by more than the 
        # tolerance value set in defaults module
        adiff_tol = sim_defaults.scale_factor_difference_tol
        adiff = np.abs(nearest_snapshot - scale_factor)
        if adiff > adiff_tol:
            msg = "Closest match to desired snapshot has a scale factor of "+str(nearest_snapshot)
            warnings.warn(msg)

        return filename_of_nearest_snapshot,nearest_snapshot

    def numptcl_to_string(self,numptcl):
        """ Reduce the input number to a 3-character string used to encode 
        the number of particles in the particle catalog filenames.

        Parameters 
        ----------
        numptcl : float or int 
            Number specifying the number of particles in the downsampled catalog.

        Returns 
        -------
        output_string : string 
            3-character string used in the filename conventions of the particle data catalogs.

        """

        # First find the order of magnitude of numptcl (there must be a more elegant way than this)
        oom_tester=False
        ipower=0
        while oom_tester==False:
            powfactor = 10.**ipower
            reduced_numptcl = np.round(numptcl/powfactor)
            if reduced_numptcl<10:
                oom_tester=True
            ipower += 1

        # Now use the above to reduce numptcl to three characters
        power = ipower-1
        powfactor = 10.**power
        reduced_numptcl = np.round(numptcl/powfactor)
        ce = str(power)
        cp = str(int(np.floor(reduced_numptcl)))
        output_string = cp+'e'+ce

        return output_string

    def load_catalog(self,catalog_type,
        dirname=None,filename=None,
        download_yn=False,url=sim_defaults.aph_web_location):
        """ Use the astropy reader to load the halo or particle catalog into memory.

        Parameters 
        ----------
        dirname : string 
            Name of directory where filename is stored.

        filename : string 
            Name of file being loaded into memory. 

        download_yn : boolean, optional
            If set to True, and if filename is not already stored in the cache directory, 
            method will attempt to download the file from the provided url. If there is no corresponding 
            file at the input url, an exception will be raised.

        url : string 
            Web location from which to download the catalog if it is not present in the cache directory.

        Returns 
        -------
        catalog : object
            Data structure located at the input filename.

        """
        if filename==None:
            if catalog_type=='subhalos':
                filename = self.default_halo_catalog_filename
            elif catalog_type=='particles':
                filename = self.default_particle_catalog_filename
            else:
                raise KeyError("Must supply catalog_type to be either "
                    "'particles' or 'subhalos'")
        if dirname==None:
            if catalog_type=='subhalos':
                dirname = self.halo_catalog_dirname
            elif catalog_type=='particles':
                dirname = self.particle_catalog_dirname
            else:
                raise KeyError("Must supply catalog_type to be either "
                    "'particles' or 'subhalos'")

        if os.path.isfile(os.path.join(dirname,filename)):
            catalog = Table.read(os.path.join(dirname,filename),path='data')
        else:
            ### Requested filename is not in cache, and external download is not requested
            if download_yn==False:
                return None
            else:
                # Download one of the default catalogs hosted at Yale
                if filename==self.default_halo_catalog_filename:
                    catalog_type='subhalos'
                if filename==self.default_particle_catalog_filename:
                    catalog_type='particles'
                else:
                    raise IOError("Input filename does not match one of the provided default catalogs")
                ###
                remote_filename = os.path.join(url,filename)
                fileobj = urllib2.urlopen(remote_filename)
                output_directory = configuration.get_catalogs_dir(catalog_type=catalog_type)
                output_filename = os.path.join(output_directory,filename)
                output = open(output_filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = fits.open(output_filename)
                catalog = Table(hdulist[1].data)

        return catalog


    def download_all_default_catalogs(self):
        """ If not already in cache, 
        download default particle and halo catalogs from Yale website.
        """

        url = sim_defaults.aph_web_location

        ### Download halo catalogs
        catalog_type = 'subhalos'
        output_directory = configuration.get_catalogs_dir(catalog_type=catalog_type)
        filename = self.default_halo_catalog_filename
        remote_filename = os.path.join(url,filename)
        if not os.path.isfile(os.path.join(output_directory,filename)):
            warnings.warn("Downloading default halo catalog")
            fileobj = urllib2.urlopen(remote_filename)
            output_filename = os.path.join(output_directory,filename)
            output = open(output_filename,'wb')
            output.write(fileobj.read())
            output.close()

        ### Download particle catalogs
        catalog_type = 'particles'
        output_directory = configuration.get_catalogs_dir(catalog_type=catalog_type)
        filename = self.default_particle_catalog_filename
        remote_filename = os.path.join(url,filename)
        if not os.path.isfile(os.path.join(output_directory,filename)):
            warnings.warn("Downloading default particle catalog")
            fileobj = urllib2.urlopen(remote_filename)
            output_filename = os.path.join(output_directory,filename)
            output = open(output_filename,'wb')
            output.write(fileobj.read())
            output.close()


###################################################################################################


###################################################################################################

