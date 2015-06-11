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

        # secondary halo property percentile key
        self.sec_haloprop_percentile_key=self.sec_haloprop_key+'_percentile'

        # add the assembly bias parameters to the param_dict so that they may 
        # be varied in an MCMC if needed.
        self.param_dict['ab_percentile']=ab_percentile
        self.param_dict['frac_dNmax']=frac_dNmax

        # check that these parameter values do not violate number conservation
        self.check_valid_ab_parameters()


    # checks validity of the input parameters
    def check_valid_ab_parameters(self):
        """
        Parameters
        ----------

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
    def assign_sec_haloprop_percentiles(self,inp_halo_catalog,num_mass_bins=35,append_mass_bins=True):
        """
        Parameters
        ----------
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
        inp_halo_catalog[self.sec_haloprop_percentile_key]=np.zeros_like(inp_halo_catalog[self.prim_haloprop_key])

        # arrange logarithmic bins on mass (or prim_haloprop_key)
        lg10_min_mass=np.log10(np.min(inp_halo_catalog[self.prim_haloprop_key]))-0.001
        lg10_max_mass=np.log10(np.max(inp_halo_catalog[self.prim_haloprop_key]))+0.001
        mass_bins=np.logspace(lg10_min_mass,lg10_max_mass,num=num_mass_bins)

        # digitize the masses so that we can access them bin-wise
        in_mass_bin=np.digitize(inp_halo_catalog[self.prim_haloprop_key],mass_bins)

        # store the mass bins for each halo so that bin membership is easily retrievable
        # if optional input argument append_mass_bins=False, then do not store the mass bin 
        # information.
        if (append_mass_bins):
            bin_key=self.prim_haloprop_key+'_bin_index'
            inp_halo_catalog[bin_key]=in_mass_bin

        # sort on secondary property only with each mass bin
        # iterating to num_mass_bins+1 ensures that any halos that exceed the 
        # maximum mass in the bins is accounted for.
        for idummy in range(num_mass_bins):
            indices_of_mass_bin=np.where(in_mass_bin==idummy)

            # Find the indices that sort by the secondary property
            ind_sorted=np.argsort(inp_halo_catalog[self.sec_haloprop_key][indices_of_mass_bin])

            percentiles=np.zeros_like(inp_halo_catalog[self.sec_haloprop_key][indices_of_mass_bin])
            percentiles[ind_sorted]=(np.arange(len(inp_halo_catalog[self.sec_haloprop_key][indices_of_mass_bin]))+1.0)/\
                float(len(inp_halo_catalog[self.sec_haloprop_key][indices_of_mass_bin]))

            #print ' Percentiles have been assigned'
            #print ' Bin number = ',idummy
            #print ' Bin edges = ', mass_bins[idummy-1], ' - ', mass_bins[idummy],'   ...'
            #print ' Number of halos in bin = ',len(indices_of_mass_bin[0])
            #print ' indices_of_mass_bin = ',indices_of_mass_bin
            #print ' masses in mass bin = ',inp_halo_catalog[self.prim_haloprop_key][indices_of_mass_bin],' \n *** \n'
            #if not any(indices_of_mass_bin[0]):
            #    print ' --- no halos make the cut here \n -------- '
            #print ' Shape of percentiles is = ',np.shape(percentiles)
            #print ' First few values are = ',percentiles[0:4],' \n'
            #print ' ------------- * ------------------ \n \n'
            
            # place the percentiles into the catalog
            inp_halo_catalog[self.sec_haloprop_percentile_key][indices_of_mass_bin]=1.0-percentiles

        return None





    # compute mean halo occupation
    def mean_occupation(self,inp_halo_catalog,append_to_catalog=False):
        """
        Parameters
        ----------
        inp_halo_catalog : astropy table
            halo catalog that can be used to assign occupation based on 
            the halo primary and secondary properties.
        
        append_to_catalog : boolean
            If true, the mean occupation for the halo will be appended to the halo catalog. 
            Default is false.

        Notes
        -----
        Method to compute mean halo occupation of halos in the halo catalog. This includes 
        assembly bias as follows. If self.param_dict['frac_dNmax']>=0, then we assume that 
        self.param_dict['ab_percentile'] refers to the highest percentile 
        and that the sign of the shift, deltangal, for those is positive. 
        Otherwise, the sign of the shift, deltangal, is negative for the highest 
        percentile. For example, if self.param_dict['frac_dNmax']=-0.4 and 
        self.param_dict['ab_precentile']=0.20, then the highest 20% of the population 
        according to sec_haloprop_key have a mean occupation that is lower than the 
        average for all halos of that mass by 0.4.
        """
    
        # get the baseline hod without any assembly bias
        num_mean_noab=self.standard_mean_occupation(halos=inp_halo_catalog)

        # get perturbation due to assembly bias. this proceeds as follows.
        # if self.param_dict['frac_dNmax']>=0, then we assume that 
        # self.param_dict['ab_percentile'] refers to the highest percentile 
        # and that the sign of the shift, deltangal, for those is positive. 
        # Otherwise, the sign of the shift, deltangal, is negative for the highest 
        # percentile. For example, if self.param_dict['frac_dNmax']=-0.4 and 
        # self.param_dict['ab_precentile']=0.20, then the highest 20% of the population 
        # according to sec_haloprop_key have a mean occupation that is lower than the 
        # average for all halos of that mass by 0.4.

        if (self.param_dict['frac_dNmax']>=0.0):
            # this is the case where the upper percentile has a positive perturbation
            delta_max_1 = 1.0 - num_mean_noab
            delta_max_2 = (1.0 - self.param_dict['ab_percentile'])*num_mean_noab/ \
                self.param_dict['ab_percentile']
            delta_n_upper_percentile=self.param_dict['frac_dNmax']*np.minimum(delta_max_1,delta_max_2)
            delta_n_lower_percentile=-self.param_dict['ab_percentile']*delta_n_upper_percentile/ \
                (1.0-self.param_dict['ab_percentile'])

        else:
            # this is the case where the upper percentile has a negative enhancement
            delta_min_1=-num_mean_noab
            delta_min_2=-(1.0-self.param_dict['ab_percentile'])*(1.0-num_mean_noab)/ \
                self.param_dict['ab_percentile']
            delta_n_upper_percentile=np.abs(self.param_dict['frac_dNmax'])* \
                np.maximum(delta_min_1,delta_min_2)
            delta_n_lower_percentile=-self.param_dict['ab_percentile']*delta_n_upper_percentile/ \
                (1.0 - self.param_dict['ab_percentile'])

        # Now we assign the shift, delta n, based upon the percentiles in the halo catalog.
        delta_num_gal=np.zeros_like(inp_halo_catalog[self.prim_haloprop_key])
        delta_num_gal=np.where(inp_halo_catalog[self.sec_haloprop_percentile_key]>=self.param_dict['ab_percentile'],
            delta_n_upper_percentile,delta_n_lower_percentile)

        num_gal=num_mean_noab+delta_num_gal

        if (append_to_catalog):
            # then append the mean occupation number to the halo catalog so that each 
            # halo knows its mean occupation.
            inp_halo_catalog['delta_num_gal']=delta_num_gal
            inp_halo_catalog['ncen_mean']=num_mean_noab

        return num_gal











    # routine to compute non-assembly biased mean occupation
    def standard_mean_occupation(self,**kwargs):
        """
        Parameters
        ----------
        keyword arguments are those of the standard model that it 
        inherits from.
        Notes
        -----
        Compute the mean occupation of halos WITHOUT assembly bias. 
        This will use the standard model instance that this is instantiated 
        with.
        """
        return self.standard_cen_model.mean_occupation(**kwargs)









