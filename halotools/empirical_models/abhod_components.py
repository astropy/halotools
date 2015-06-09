# -*- coding: utf-8 -*-
"""
Attempt to develop simple HOD models that include assembly bias.
"""

import numpy as np # numpy as usual
import hod_components # Andrew Hearin's HOD components
import model_defaults # default parameters for HOD models

# A class to implement 2-population, heaviside assembly bias
class HeavisideCenAssemBiasModel(hod_components.OccupationComponent):
    """
    Parameters
    ----------
    standard_cen_model : OccupationComponent
        an instance of a standard central galaxy occupation component that implements an HOD with no assembly bias

    ab_percentile : float
        percentile at which to implement heavside 2-population assembly bias

    frac_dNmax : float
        fraction of maximal assembly bias effect

    secondary_haloprop_key : string
        the secondary halo property upon which assembly bias is based

    gal_type : string, optional
        Name of the galaxy population being modeled, e.g., ``cens`. 
        Default setting is ``centrals``. 

    Notes
    -----
    This defines a class to deal with the assembly bias piece of the HOD, if present.
    In this iteration, I am attempting to sub-class it to OccupationComponent.
    """

    def __init__(self,
        standard_cen_model,
        ab_percentile=0.50,
        frac_dNmax=0.0,
        secondary_haloprop_key='vmax'):
        
        """
        Parameters
        ----------
        standard_cen_model : OccupationComponent
            an instance of a standard central galaxy occupation component that implements an HOD with no assembly bias

        ab_percentile : float
            percentile at which to implement heavside 2-population assembly bias

        frad_dNmax : float
            fraction of maximal assembly bias effect

        secondary_haloprop_key : string
            the secondary halo property upon which assembly bias is based

        gal_type : string, optional
            Name of the galaxy population being modeled, e.g., ``cens`. 
            Default setting is ``centrals``. 

        Notes
        -----
        This defines a class to deal with the assembly bias piece of the HOD, if present.
        In this iteration, I am attempting to sub-class it to OccupationComponent.
        """
        
        # establish the non-assembly biased model that these routines will
        # be working with.
        self.standard_cen_model=standard_cen_model

        # call super-class init routine
        # the instance inherits the basic model of the non-assembly biased model to which it is tied.

        #hod_components.OccupationComponent.__init__(self,
        super(HeavisideCenAssemBiasModel,self).__init__(
            standard_cen_model.gal_type,
            standard_cen_model.threshold,
            standard_cen_model.occupation_bound,
            prim_haloprop_key=standard_cen_model.prim_haloprop_key,
            sec_haloprop_key=secondary_haloprop_key,
            param_dict=standard_cen_model.param_dict)

        # add the assembly bias parameters to the param_dict so that they may 
        # be varied in an MCMC if needed.
        self.param_dict['ab_percentile']=ab_percentile
        self.param_dict['frac_dNmax']=frac_dNmax

        print self.param_dict

        # check that these parameter values do not violate number conservation
        self.check_valid_ab_parameters()


    # checks validity of the input parameters
    def check_valid_ab_parameters(self):
        """
        Parameters
        ----------
        self

        Notes
        -----
        Checks if the assembly bias parameters are valid. 
        In particular, this checks that the assembly bias percentile lies 
        between 0 and 1 and that the fraction of the maximum effect lies 
        between -1.0 and 1.0.
        """
        # percentile must be between 0.0 and 1.0
        if (self.param_dict['ab_percentile'] < 0.0):
            self.param_dict['ab_percentile'] = 0.0
        elif (self.param_dict['ab_percentile']>1.0):
            self.param_dict['ab_percentile']=1.0
        
        # fraction of maximum effect must have magnitude <= 1.0
        if (self.param_dict['frac_dNmax']<-1.0):
            self.param_dict['frac_dNmax']=-1.0
        elif (self.param_dict['frac_dNmax']>1.0):
            self.param_dict['frac_dNmax']=1.0

        return None





    # assign halos percentile values of the secondary halo property
    def assign_halo_secondary_percentiles(self,inp_halo_catalog,num_mass_bins=35):
        """
        Parameters
        ----------
        self

        inp_halo_catalog : astropy table 
            stores halo catalog being used to make mock galaxy population

        num_mass_bins : integer
            number of bins of mass within which to assign secondary property percentiles

        Notes
        -----
        Takes the input halo catalog and uses the assembly bias model to assign percentiles 
        of the property in sec_haloprop_key to each halo.
        """
        
        # new halo property
        sec_haloprop_percentile_key=self.sec_haloprop_key+'_percentile'
        inp_halo_catalog[sec_haloprop_percentile_key]=np.zeros_like(inp_halo_catalog[self.prim_haloprop_key])

        # arrange logarithmic bins on mass (or prim_haloprop_key)
        lg10_min_mass=np.log10(np.min(inp_halo_catalog[self.prim_haloprop_key]))-0.001
        lg10_max_mass=np.log10(np.max(inp_halo_catalog[self.prim_haloprop_key]))+0.001
        mass_bins=np.logspace(lg10_min_mass,lg10_max_mass,num=num_mass_bins+1)

        # digitize the masses so that we can access them bin-wise
        in_mass_bin=np.digitize(inp_halo_catalog[self.prim_haloprop_key],mass_bins)

        # store the mass bins for each halo so that bin membership is easily retrievable
        bin_key=self.prim_haloprop_key+'_bin_index'
        inp_halo_catalog[bin_key]=in_mass_bin

        # sort on secondary property only with each mass bin
        







    # compute mean halo occupation
    def mean_occupation(self):
        """
        Method to compute mean halo occupation of halos in the halo catalog.
        """
        pass

    # routine to compute non-assembly biased mean occupation
    def standard_mean_occupation(self,*args):
        """
        Compute the mean occupation of halos WITHOUT assembly bias. 
        This will use the standard model instance that this is instantiated 
        with.
        """
        return self.standard_cen_model.mean_occupation(*args)









