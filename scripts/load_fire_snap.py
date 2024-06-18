import numpy as np
import h5py
import os
import ctypes
from gizmopy.load_from_snapshot import *
# This file was written by Phil Hopkins (Caltech) for FIRE/GIZMO simulations.


## return stellar luminosity [in solar] for given band, age [in Gyr], and metallicity of population
def stellar_luminosity(band_name, stellar_mass_in_solar, age_in_gyr, total_metallicity_massfraction, 
        imf='chabrier', quiet=True, **kwargs):
    ''' Load stellar luminosity in specified band. 
          band_name: 'Bol','U','B','V','R','I','J','H','K','u','g','r','i','z' : desired band
          stellar_mass_in_solar: list of masses of elements to return luminosities for
          age_in_gyr: list of ages of elements to return luminosities for
          total_metallicity_massfraction: list of metallicities (Z, in mass-fraction units) of elements
          
          Optional:
          imf='chabrier' (default), or 'salpeter' for salpeter IMF
          quiet=False (default), or 'True' for no output
          any additional parameters which can be passed to 'colors_table' in pfh_utils 
    '''          
    import pfh_utils as util
    # convert from our nice band-names to the annoying list convention of the subroutine
    bands={"Bol":0, "U":1, "B":2, "V":3, "R":4, "I":5, "J":6, "H":7, "K":8, "u":9, "g":10, "r":11, "i":12, "z":13}
    band_id = 0; # default bolometric
    if(band_name in bands): band_id = bands[band_name]
    salpeter_key=0; chabrier_key=1; # default to chabrier
    if(imf=='salpeter'): 
        salpeter_key=1; chabrier_key=0;
    quiet_key=0
    if(quiet): quiet_key=1
    ## get the light-to-mass ratios from our subroutine
    l_over_m = util.colors_table( np.array(age_in_gyr), np.array(total_metallicity_massfraction)/0.02, 
        BAND_ID=band_id, SALPETER_IMF=salpeter_key, CHABRIER_IMF=chabrier_key, 
        QUIET=quiet_key, **kwargs)
    return np.array(stellar_mass_in_solar) * l_over_m 


## return exact solution for lookback time for a flat universe
def lookback_time_Gyr_flatuniverse(scale_factor, h=0.71, Omega_M=0.27):
    ''' Return lookback time in Gyr to specified scale-factor, with optionally-specified
         hubble constant and omega matter, for a flat Lambda-CDM universe '''
    x = Omega_M/(1.-Omega_M) / (scale_factor*scale_factor*scale_factor);
    t = 13.777 * (0.71/h) * (2./(3.*np.sqrt(1.-Omega_M))) * np.log( np.sqrt(x) / (-1. + np.sqrt(1.+x)) );
    return t;


## Lookback time (in context of converting stellar ages)
def get_stellar_age_gyr(scale_factor_form, scale_factor=1., h=0.71, Omega_M=0.27):
    ''' Return stellar age in Gyr given their scale factor of formation, the current
          scale factor (default=1), hubble constant and omega matter, assuming 
          a flat Lambda-CDM cosmology '''
    a0 = scale_factor_form; a2 = scale_factor; 
    x0 = (Omega_M/(1.-Omega_M))/(a0*a0*a0); x2 = (Omega_M/(1.-Omega_M))/(a2*a2*a2);
    age = 13.777 * (0.71/h) * (2./(3.*np.sqrt(1-Omega_M))) * np.log(np.sqrt(x0*x2)/((np.sqrt(1.+x2)-1.)*(np.sqrt(1.+x0)+1.)));
    return age


## return temperature from internal energy, helium fraction, electron abundance, total metallicity, and density
def get_temperature(internal_egy_code, helium_mass_fraction, electron_abundance, total_metallicity, mass_density, f_neutral=np.zeros(0), f_molec=np.zeros(0), key='Temperature'):
    ''' Return estimated gas temperature, given code-units internal energy, helium 
         mass fraction, and electron abundance (number of free electrons per H nucleus). 
         this will use a simply approximation to the iterative in-code solution for 
         molecular gas, total metallicity, etc. so does not perfectly agree 
         (but it is a few lines, as opposed to hundreds) '''
    internal_egy_cgs=internal_egy_code*1.e10; gamma_EOS=5./3.; kB=1.38e-16; m_proton=1.67e-24; X0=0.76;
    total_metallicity[(total_metallicity>0.25)] = 0.25;
    helium_mass_fraction[(helium_mass_fraction>0.35)] = 0.35;
    y_helium = helium_mass_fraction / (4.*(1.-helium_mass_fraction));
    X_hydrogen = 1. - (helium_mass_fraction+total_metallicity);
    T_mol = 100. * (mass_density*4.04621);
    T_mol[(T_mol>8000.)] = 8000.;
    A0 = m_proton * (gamma_EOS-1.) * internal_egy_cgs / kB;
    mu = (1. + 4.*y_helium) / (1.+y_helium+electron_abundance); 
    X=X_hydrogen; Y=helium_mass_fraction; Z=total_metallicity; nel=electron_abundance; 
    
    if(np.array(f_molec).size > 0):
        fmol = 1.*f_molec;
        fH=X_hydrogen; f=fmol; xe=nel;
        gamma_eff = 1. + (fH*((1.-f)/1. + f/2.) + (1.-fH)/4.) / (fH*((1.-f + xe)/(1.*(5./3.-1.)) + f/(2.*(7./5.-1.))) + (1.-fH)/(4.*(5./3.-1.))); ## assume He is atomic, H has a mass fraction f molecular
        A0 = m_proton * (gamma_eff-1.) * internal_egy_cgs / kB;
    else:
        fmol=0;    
        if(2==2):
            for i in range(3):
                mu = 1. / ( X*(1.-0.5*fmol) + Y/4. + nel*X0 + Z/(16.+12.*fmol));
                T = mu * A0 / T_mol; 
                fmol = 1. / (1. + T*T);
    mu = 1. / ( X*(1.-0.5*fmol) + Y/4. + nel*X0 + Z/(16.+12.*fmol));
    T = mu * A0;
    if('Temp' in key or 'temp' in key): return T;
    if('Weight' in key or 'weight' in key): return mu;
    if('Number' in key or 'number' in key): return mass_density * 404.621 / mu;    
    return T;


## return star particle mass at formation time relative to present mass (based on average mass-loss rates, mean correction)
def get_stellar_formation_mass(mass, age_in_gyr):
    ''' Use a simple fitting function to return the at-formation stellar mass 
         of a star particle, based on its present-day age in Gyr and current mass'''
    fac=0.62*(age_in_gyr**(-0.071)); fac[fac>1]=1; fac[fac<0.8]=0.8;
    return mass/fac;


## return X-ray thermal bremstrahhlung luminosity of a particle. DOES NOT INCLUDE metal lines 
def gas_xray_brems(m, rho, T, x_e, x_h):
    ''' Estimated X-ray cooling luminosity from thermal bremstrahhlung in erg/sec.
         Assumes partially-ionized gas, usual cosmo units (M in 1e10 Msun, L kpc), 
         given T in K, x_e/x_h electron/neutral fractions. Returns 
         luminosity above ~0.5keV '''
    T_keV=8.617e-8*T; mu_wt=1.316/(1.079+x_e); n_e=rho*404.621/mu_wt; N_normed=m*1.427e43;
    return N_normed * np.sqrt(T_keV)*np.exp(-0.5/T_keV) * n_e*x_e*(1.-x_h)


## return H-alpha luminosity assuming simple case-B recombinations
def gas_Halpha_lum(m, rho, T, x_e, x_h):
    ''' Estimated H-alpha recombination luminosity (case-B) in erg/sec.
         Usual cosmo units (M in 1e10 Msun, L kpc, T in K, etc)'''
    n_p=0.76*(rho/404.62)*(1.-x_h); N_e=x_e*0.76*(m*1.189e67); a_B=2.1e-13/np.sqrt(1000.+T); 
    return 1.37e-12*N_e*n_p*a_B;


## return CO luminosity [in K km/s pc^2] in very approximate order-of-magnitude guess
def gas_CO_lum(m, rho, T, x_h, alpha_CO=4.6, nh_thold=5., T_thold=5000.):
    ''' Estimated CO luminosity in [K km/s pc^2]
        assuming Milky Way-like alpha-CO (ULIRG-like gives 5x larger), and simple 
        density-temperature cut for molecular gas. Can be very, very wrong!'''
    nh=rho*406.4; m_mol=(1.-x_h)*m*1.e10*np.exp(-T/T_thold-nh_thold/nh); 
    return m_mol/alpha_CO;


## return dust temperature [in K] in approximate manner based on the saved radiation field
##   [if saved] estimated with standard FIRE modules at the location of the cell, plus CMB
def dust_temperature_firedefault(e_rad, z_redshift):
    ''' Estimated dust temperature in [K], assuming a simple three-component 
        mix of incident radiation followed in-code, including CMB (with the 
        appropriate redshift-dependent properties), IR radiation at a temperature 
        equal to MIN(30,T_cmb), and higher-energy (NIR/optical/UV/ionizing/XR) 
        radiation with an effective radiation temperature of 5800 K. These 
        plus the radiation energy densities give the local dust temperature 
        estimator, assuming a beta=1 opacity law in the relevant IR regime.
    '''
    zplus=1.+z_redshift; T_cmb=2.73*zplus; e_CMB_eV=0.26*zplus*zplus*zplus*zplus;
    T_dust_ext=30.; T_hiegy=5800.;
    if(T_cmb>T_dust_ext): T_dust_ext=T_cmb;
    e_unit_to_eV = 4.22; # convert code units to eV/cm^3
    e_hiegy = (e_rad[:,0]+e_rad[:,1]) * e_unit_to_eV
    e_IR = e_rad[:,2] * e_unit_to_eV    
    T_eff=2.92 * ((T_cmb*e_CMB_eV + T_dust_ext*e_IR + T_hiegy*e_hiegy)**(0.2))
    T_eff[(T_eff>2000.)]=2000.; # limit at sublimation temperature
    T_eff[(T_eff<1.)]=1.; # limit at floor where opacity law breaks down
    return T_eff
    

## subroutine to load quantities from FIRE snapshots -- makes assumptions about units, etc, 
##   which are not necessarily generic, but specific for our simulations
def load_fire_snap(value, ptype, sdir, snum, **kwargs):
    ''' 
    Routine to load different quantities from snapshots. Automagically attempts
    to handle many things for you, like where the data is in the file (header, 
    type, etc), formatting, name and numbering conventions for snapshots, 
    concatenating outputs of multi-part snaps, determining if the snapshot is 
    cosmological and converting to physical units, and more.
    
    'load_from_snapshot' is the core sub-routine here. you can call this just 
    like you would call that, and you should read the help for that routine to 
    see the options. all keywords, etc, for that work here. 
    
    the only difference is this routine is value-added, by adding a number of 
    custom 'Value' calls that allow them to be treated just like any other raw
    data actually saved in the snapshots. 
    
    Examples include:
      'Temperature': (gas temperature in K loaded directly, instead of internal energy)
      'Z': (total metal mass fraction, alone, not the full metallicity block)
      'StellarAgeGyr': (star particle age in Gyr, at the time of the snapshot)
      'StellarFormationMasses': (mass of star particle in code units when it formed)
      'StellarLuminosity_X': (observed luminosity in any of ~14 bands, set 
          X = Bol,U,V,B,R,I,J,H,K,u,g,r,i,z  here to get that band)
      'SmoothingLength': for Type=0 (gas), saved value. For any others, 
          will check if saved SmoothingLength file exists, or else generate one 
          based on neighbors of that type
      'Density': returns mass density of neighbors, for any type like 'SmoothingLength'
      'SpecificStarFormationRate': return gas specific SFR (=SFR/M)
      'GasLuminosity_Xray': (approximate) gas thermal bremstrahhlung X-ray luminosity
      'GasLuminosity_CO': (approximate) CO luminosity using standard alpha-CO and a 
          simple temperature-density based estimate of the molecular gas mass
      'GasLuminosity_Halpha': H-alpha luminosity using ionized 
          fraction and standard expressions for recombination rates
    '''

    ## add various special flags here for additional quantities of common use
    
    # gas temperature [as opposed to internal energy]
    if(value=='Temperature' or value=='MeanMolecularWeight' or value=='NumberDensity'or value=='Number_Density'):
        if(ptype != 0): return np.zeros(0);
        rho = load_from_snapshot('Density',0,sdir,snum,**kwargs)
        u = load_from_snapshot('InternalEnergy',0,sdir,snum,**kwargs)
        n_elec = load_from_snapshot('ElectronAbundance',0,sdir,snum,**kwargs)
        z_tot = load_from_snapshot('Metallicity',0,sdir,snum,axis_mask=0,**kwargs)
        z_he = load_from_snapshot('Metallicity',0,sdir,snum,axis_mask=1,**kwargs)
        keys = load_from_snapshot('Keys',0,sdir,snum,**kwargs)
        f_neutral = np.zeros(0)
        f_molec = np.zeros(0)
        if('NeutralHydrogenAbundance' in keys): f_neutral = load_from_snapshot('NeutralHydrogenAbundance',0,sdir,snum,**kwargs)
        if('MolecularMassFraction' in keys): f_molec = load_from_snapshot('MolecularMassFraction',0,sdir,snum,**kwargs)
        return get_temperature(u, z_he, n_elec, z_tot, rho, f_neutral=f_neutral, f_molec=f_molec, key=value)
        
    # total gas metallicity [as opposed to the giant all-metals array]
    if(value=='Z'):
        return load_from_snapshot('Metallicity',ptype,sdir,snum,axis_mask=0,**kwargs)
        
    # stellar age in physical units [as opposed to formation time in scale factor]
    if(value=='StellarAgeGyr' or value=='Stellar_Age_Gyr'):
        if(ptype == 0): return np.zeros(0);
        tform = load_from_snapshot('StellarFormationTime',4,sdir,snum,**kwargs)
        time = load_from_snapshot('Time',-1,sdir,snum,**kwargs)
        if(evaluate_if_cosmological(sdir,snum,**kwargs)):
            hubble = load_from_snapshot('HubbleParam',-1,sdir,snum,**kwargs)
            keys = load_from_snapshot('keys',-1,sdir,snum,**kwargs)
            omega = 1.0;
            if('Omega0' in keys): omega = load_from_snapshot('Omega0',-1,sdir,snum,**kwargs)
            if('OmegaMatter' in keys): omega = load_from_snapshot('OmegaMatter',-1,sdir,snum,**kwargs)
            if('Omega_Matter' in keys): omega = load_from_snapshot('Omega_Matter',-1,sdir,snum,**kwargs)
            return get_stellar_age_gyr(tform,scale_factor=time,h=hubble,Omega_M=omega)
        return time-tform

    # star formation time for non-type-4 particles (common in non-cosmological sims)
    if(value=='StellarFormationTime' or value=='Stellar_Formation_Time'):
        if(ptype == 4): return load_from_snapshot('StellarFormationTime',4,sdir,snum,**kwargs)
        m = load_from_snapshot('Masses',ptype,sdir,snum,**kwargs)
        if(ptype == 2): return 0.1 + 10.*np.random.rand(m.size)
        return 10. + 4.*np.random.rand(m.size)
        
    # initial mass of star particles [at formation time]
    if(value=='StellarFormationMasses' or value=='Stellar_Formation_Masses'):
        if(ptype != 4): return np.zeros(0);
        m = load_from_snapshot('Masses',4,sdir,snum,**kwargs)
        age = load_fire_snap('StellarAgeGyr',4,sdir,snum,**kwargs)
        return get_stellar_formation_mass(m,age)

    # stellar luminosities in different bands, specified as 'StellarLuminosity_BAND' where BAND=Bol,U,B,V,R,I,J,H,K,u,g,r,i,z
    if('StellarLuminosity' in value or 'Stellar_Luminosity' in value):
        if(ptype != 4): return np.zeros(0);
        m = load_from_snapshot('Masses',4,sdir,snum,**kwargs) * 1.e10 # convert to solar
        age = load_fire_snap('StellarAgeGyr',4,sdir,snum,**kwargs)
        z = load_fire_snap('Z',4,sdir,snum,**kwargs)
        return stellar_luminosity(value.split('_')[-1],m,age,z,quiet=True)

    # radius of compact support of resolution elements of given type, e.g. 'smoothing length'
    if('Smoothing_Length' in value or 'SmoothingLength' in value or 'Hsml' in value):
        if(ptype == 0): 
            return load_from_snapshot('SmoothingLength',ptype,sdir,snum,**kwargs)
        else:
            return load_particle_hsml(ptype,sdir,snum,**kwargs)

    # mass density of given species 
    if(value=='Density' or value=='Densities'):
        if(ptype == 0): return load_from_snapshot('Density',ptype,sdir,snum,**kwargs)
        m = load_fire_snap('Masses',ptype,sdir,snum,**kwargs)
        h = load_fire_snap('SmoothingLength',ptype,sdir,snum,**kwargs)
        return m/(0.5081*h)**3 # this converts from the kernel radius of compact support to density for NNgb=32

    # specific star formation rate
    if(value=='SpecificSFR' or value=='SpecificStarFormationRate' or value=='SSFR'):
        if(ptype != 0): return np.zeros(0);
        sfr = load_from_snapshot('StarFormationRate',4,sdir,snum,**kwargs)
        m = load_from_snapshot('Masses',4,sdir,snum,**kwargs)
        return sfr/m

    # X-ray luminosity of gas 
    if(value=='GasLuminosity_Xray' or value=='XR' or value=='Xray' or value=='X-ray' or value=='Luminosity_Xray' or value=='XrayLuminosity'):
        if(ptype != 0): return np.zeros(0);
        T = load_fire_snap('Temperature',ptype,sdir,snum,**kwargs)
        m = load_fire_snap('Masses',ptype,sdir,snum,**kwargs)
        rho = load_fire_snap('Density',ptype,sdir,snum,**kwargs)
        x_e = load_from_snapshot('ElectronAbundance',ptype,sdir,snum,**kwargs)
        x_h = load_from_snapshot('NeutralHydrogenAbundance',ptype,sdir,snum,**kwargs)
        return gas_xray_brems(m, rho, T, x_e, x_h)
    
    # H-alpha luminosity of gas
    if(value=='Halpha' or value=='H-alpha' or value=='GasLuminosity_Halpha'):
        if(ptype != 0): return np.zeros(0);
        T = load_fire_snap('Temperature',ptype,sdir,snum,**kwargs)
        m = load_fire_snap('Masses',ptype,sdir,snum,**kwargs)
        rho = load_fire_snap('Density',ptype,sdir,snum,**kwargs)
        x_e = load_from_snapshot('ElectronAbundance',ptype,sdir,snum,**kwargs)
        x_h = load_from_snapshot('NeutralHydrogenAbundance',ptype,sdir,snum,**kwargs)
        return gas_Halpha_lum(m, rho, T, x_e, x_h)

    # CO luminosity of gas
    if(value=='CO' or value=='GasLuminosity_CO'):
        if(ptype != 0): return np.zeros(0);
        T = load_fire_snap('Temperature',ptype,sdir,snum,**kwargs)
        m = load_fire_snap('Masses',ptype,sdir,snum,**kwargs)
        rho = load_fire_snap('Density',ptype,sdir,snum,**kwargs)
        x_h = load_from_snapshot('NeutralHydrogenAbundance',ptype,sdir,snum,**kwargs)
        return gas_CO_lum(m, rho, T, x_h)

    # Dust temperature (approximated as in-code for non-RHD-IR runs)
    if('Dust_Temp' in value or 'DustTemp' in value):
        if(ptype != 0): return np.zeros(0)
        z_redshift = 0.0
        if(evaluate_if_cosmological(sdir,snum,**kwargs)):
            z_redshift = load_from_snapshot('Redshift',-1,sdir,snum,**kwargs)
        e_rad = load_fire_snap('PhotonEnergy',ptype,sdir,snum,**kwargs) # includes 'h'
        m = load_fire_snap('Masses',ptype,sdir,snum,**kwargs) # includes 'h'
        rho = load_fire_snap('Density',ptype,sdir,snum,**kwargs) # includes 'h'
        vol_inv = rho / m
        e_rad = (e_rad.transpose() * vol_inv).transpose(); ## convert to volumetric radiation energy density 
        return dust_temperature_firedefault(e_rad, z_redshift)
        
    if(value=='B' or 'Bfield' in value or 'BField' in value or 'Magneticfield' in value or 'MagneticField' in value or 'Magnetic_field' in value or 'Magnetic_Field' in value):
        if(ptype != 0): return np.zeros(0)
        return load_from_snapshot('MagneticField',ptype,sdir,snum,**kwargs)
    
    if(value=='Bmag' or value=='B_mag' or value=='magnitude_B'):
        if(ptype != 0): return np.zeros(0)
        B=load_from_snapshot('MagneticField',ptype,sdir,snum,**kwargs)
        return np.sqrt(np.sum(B*B,axis=1))

    if(value=='Volume' or value=='Volumes'):
        if(ptype != 0): return np.zeros(0)
        m = load_fire_snap('Masses',ptype,sdir,snum,**kwargs) # includes 'h'
        rho = load_fire_snap('Density',ptype,sdir,snum,**kwargs) # includes 'h'
        return m/rho

    if('MagneticEnergy' in value):
        if(ptype != 0): return np.zeros(0)
        B=load_from_snapshot('MagneticField',ptype,sdir,snum,**kwargs)
        return np.sum(B*B,axis=1) / (8.*np.pi)

    # if not a custom-coded value as the above, just pull directly from the file
    return load_from_snapshot(value,ptype,sdir,snum,**kwargs)


def evaluate_if_cosmological(sdir,snum,**kwargs):
    ''' guess if snapshot is actually cosmological or not for purposes of correcting stellar ages'''
    if(load_fire_snap('HubbleParam',0,sdir,snum,**kwargs)==1): return False
    if(numpy.abs(load_fire_snap('Time',0,sdir,snum,**kwargs)*(1.+load_fire_snap('Redshift',0,sdir,snum,**kwargs))-1.) < 1.e-6): return True
    return False
    

def snap_ext(snum,four_char=True):
    ''' 
    returns snapshot extension in string format per our default rules 
    '''
    ext='00'+str(snum);
    if(snum>=10): ext='0'+str(snum)
    if(snum>=100): ext=str(snum)
    if(four_char==True): ext='0'+ext
    if(snum>=1000): ext=str(snum)
    return ext;


def load_particle_hsml(ptype,sdir,snum,hsml_dir='',DesNgb=32,particle_mask=numpy.zeros(0),
        filename_prefix='hsmlfile',suffix='.hdf5',snap_shortname='snap',four_char=True,**kwargs):
    ''' 
    generates and saves, or (if possible) returns pre-saved non-gas Hsml values 
    '''    
    exts=snap_ext(snum,four_char=four_char);
    parent_dir=sdir;
    if(len(hsml_dir)): parent_dir=hsml_dir;

    hsmlfile = parent_dir  + '/' + filename_prefix + '_' + snap_shortname + '_' + exts + suffix
    dataset_name = 'PartType{}/DesNgb{}_SmoothingLength'.format(ptype, DesNgb)

    h = None
    if os.path.exists(hsmlfile): ## it exists! check if we have the right ptype + desngb combination
        with h5py.File(hsmlfile,'r') as F:
            if dataset_name in F:
                h=np.array(F[dataset_name][:])
                print("-- loaded smoothing lengths from dataset {} in {}".format(
                    dataset_name, hsmlfile))

    if h is None: ## no pre-computed dataset, need to do it ourselves
        xyz = load_from_snapshot('Coordinates',ptype,sdir,snum,**kwargs); # load coordinates
        h = get_particle_hsml(xyz[:,0],xyz[:,1],xyz[:,2],DesNgb=DesNgb); # do the computation
        
        # try to store the result to speed up for future runs
        if(h.size>0):
            try:
                with h5py.File(hsmlfile,'a') as F:
                    F.create_dataset(dataset_name, data=h); 
                    if 'SnapshotDirectory' in F:
                        assert F['SnapshotDirectory'][()].rstrip('/') == sdir.rstrip('/')
                        assert F['SnapshotNumber'][()] == snum
                    else:
                        F.create_dataset("SnapshotDirectory",data=sdir); 
                        F.create_dataset("SnapshotNumber",data=snum); 
                print("saved smoothing lengths to {} (in dataset {})".format(
                    hsmlfile, dataset_name))
            except IOError as err:
                print("-- Failed to save stellar smoothing lengths due to an IO Error" +
                 " of\n\t{}\n-- Will continue anyway, but you won't see a speedup" +
                 " in future runs".format(err))
    particle_mask=np.array(particle_mask)
    if(particle_mask.size > 0): 
        h=h.take(particle_mask,axis=0)
    return h;


def get_particle_hsml( x, y, z, DesNgb=32, Hmax=0.):
    '''
    subroutine which computes kernel smoothing length in post-processing 
    for a target particle list by calling the 'starhsml' shared library
    '''
    x=fcor(x); y=fcor(y); z=fcor(z); N=checklen(x); 
    ok=(ok_scan(x) & ok_scan(y) & ok_scan(z)); x=x[ok]; y=y[ok]; z=z[ok];
    if(Hmax==0.):
        dx=np.max(x)-np.min(x); dy=np.max(y)-np.min(y); dz=np.max(z)-np.min(z); ddx=np.max([dx,dy,dz]); 
        Hmax=1000.*ddx*(np.float(N)**(-1./3.)); ## mean inter-particle spacing
    ## load the routine we need
    import pfh_utils as util
    exec_call=util.return_python_routines_cdir()+'/StellarHsml/starhsml.so'
    h_routine=ctypes.cdll[exec_call];
    h_out_cast=ctypes.c_float*N; H_OUT=h_out_cast();
    ## main call to the hsml-finding routine
    h_routine.stellarhsml( ctypes.c_int(N), \
        vfloat(x), vfloat(y), vfloat(z), ctypes.c_int(DesNgb), \
        ctypes.c_float(Hmax), ctypes.byref(H_OUT) )
    ## now put the output arrays into a useful format 
    h = np.ctypeslib.as_array(np.copy(H_OUT));
    return h;


def fcor(x):
    '''subroutine which verifies type-casting as float, used to ensure proper transmission to external C'''
    return np.array(x,dtype='f',ndmin=1)
    
    
def vfloat(x):
    '''subroutine which verifies type-casting as float for vector, used to ensure proper transmission to external C'''
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float));
    
    
def ok_scan(input,xmax=1.0e10,pos=0):
    '''subroutine which removes nans and extremal values, used to prevent passing wrongly-sized transmission to external C'''
    if (pos==1):
        return (np.isnan(input)==False) & (np.abs(input)<=xmax) & (input > 0.);
    else:
        return (np.isnan(input)==False) & (np.abs(input)<=xmax);
        
        
def checklen(x):
    '''get element length [which may not be a proper array]'''
    return len(np.array(x,ndmin=1));
