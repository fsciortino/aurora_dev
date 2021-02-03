import numpy as np, sys
import matplotlib.pyplot as plt
from scipy.constants import e as q_electron,k as k_B, h, m_p, c as c_speed
plt.ion()
import periodictable
from scipy.interpolate import interp1d
import aurora

from IPython import embed

def plot_adas_spectrum(adf15_filepath, ion, ne_cm3, Te_eV, n0_cm3=0.0,
                       ion_exc_rec_dens=None, ax=None):
    '''Plot spectrum based on the lines contained in an ADAS ADF15 file
    at specific values of electron density and temperature. Charge state densities
    can be given explicitely, or alternatively charge state fractions will be automatically 
    computed from ionization equilibrium (no transport). 

    Parameters
    ----------
    adf15_filepath : str
        Path on disk to the ADAS ADF15 file of interest. All wavelengths and radiating
        components will be read. 
    ion : str
        Atomic symbol of ion of interest, e.g. 'Ar'
    ne_cm3 : float
        Local value of electron density, in units of :math:`cm^{-3}`.
    Te_eV : float
        Local value of electron temperature, in units of :math:`eV`.
    n0_cm3 : float, optional
        Local density of atomic neutral hydrogen isotopes. This is only used if the provided
        ADF15 file contains charge exchange contributions.
    ion_exc_rec_dens : list of 3 floats or None
        Density of ionizing, excited and recombining charge states that may contribute to 
        emission from the given ADF15 file. If left to None, ionization equilibrium is assumed.
    ax : matplotlib Axes instance
        Axes to plot on. If left to None, a new figure is created.

    Returns
    -------
    ax : matplotlib Axes instance
        Axes on which the plot is returned.

    Notes
    -----
    Including ionizing, excited and recombining charge states allows for a complete description
    of spectral lines that may derive from various atomic processes in a plasma.
    '''
    # ensure input ne,Te,n0 are floats
    ne_cm3=float(ne_cm3)
    Te_eV=float(Te_eV)
    n0_cm3=float(n0_cm3)
    
    # read ADF15 file
    pec_dict = aurora.read_adf15(adf15_filepath)

    # get charge state from file name -- assumes standard nomenclature, {classifier}#{ion}{charge}.dat
    cs = adf15_filepath.split('#')[-1].split('.dat')[0]
    
    # import here to avoid issues when building docs or package
    from omfit_commonclasses.utils_math import atomic_element
    
    # get nuclear charge Z and atomic mass number A
    out = atomic_element(symbol=ion)
    spec = list(out.keys())[0]
    ion_Z = int(out[spec]['Z'])
    ion_A = int(out[spec]['A'])

    if ion_exc_rec_dens is None: 
        # use ionization equilibrium fractional abundances as densities

        # get charge state distributions from ionization equilibrium
        files = ['scd','acd','ccd']
        atom_data = aurora.atomic.get_atom_data(ion,files)

        # always include charge exchange, although n0_cm3 may be 0
        logTe, fz, rates = aurora.get_frac_abundances(
            atom_data, np.array([ne_cm3,]), np.array([Te_eV,]), n0_by_ne=np.array([n0_cm3/ne_cm3,]), include_cx=True, plot=False)
        ion_exc_rec_dens = [fz[0][-4], fz[0][-3], fz[0][-2]]


    wave_A = np.zeros((len(list(pec_dict.keys()))))
    pec_ion = np.zeros((len(list(pec_dict.keys()))))
    pec_exc = np.zeros((len(list(pec_dict.keys()))))
    pec_rec = np.zeros((len(list(pec_dict.keys()))))
    pec_cx = np.zeros((len(list(pec_dict.keys()))))
    pec_dr = np.zeros((len(list(pec_dict.keys()))))
    for ii,lam in enumerate(pec_dict):
        wave_A[ii] = lam
        if 'ioniz' in pec_dict[lam]:
            pec_ion[ii] = pec_dict[lam]['ioniz'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'excit' in pec_dict[lam]:
            pec_exc[ii] = pec_dict[lam]['excit'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'recom' in pec_dict[lam]:
            pec_rec[ii] = pec_dict[lam]['recom'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'chexc' in pec_dict[lam]:
            pec_cx[ii] = pec_dict[lam]['checx'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'drsat' in pec_dict[lam]:
            pec_dr[ii] = pec_dict[lam]['drsat'].ev(np.log10(ne_cm3),np.log10(Te_eV))

    #wave_m = wave_A*1e-10
    
    # Doppler broadening: Loch's thesis equations 1.25, 1.26
    mass = m_p * ion_A
    dnu_g = np.sqrt(2.*(Te_eV*q_electron)/mass)*(c_speed/wave_A)/c_speed
    
    # set a variable delta lambda based on the width of the broadening
    dlam_A = wave_A**2/c_speed* dnu_g * 5 # 5 standard deviations
    
    lams_profs_A =np.linspace(wave_A-dlam_A, wave_A + dlam_A, 100, axis=1) 
    
    theta_tmp = 1./(np.sqrt(np.pi)*dnu_g[:,None])*\
                np.exp(-((c_speed/lams_profs_A-c_speed/wave_A[:,None])/dnu_g[:,None])**2)

    # Normalize Gaussian profile
    theta = np.einsum('ij,i->ij', theta_tmp, 1./np.trapz(theta_tmp,x=lams_profs_A,axis=1))
    
    wave_final_A = np.linspace(np.min(lams_profs_A), np.max(lams_profs_A), 100000)
    
    # contributions to spectrum
    spec_ion = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        spec_ion += interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[0]*pec_ion[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
    spec_exc = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        spec_exc += interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[1]*pec_exc[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
    spec_rec = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        spec_rec += interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[2]*pec_rec[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
    spec_dr = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        spec_dr += interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[2]*pec_dr[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
    spec_cx = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        spec_cx += interp1d(lams_profs_A[ii,:], n0_cm3*ion_exc_rec_dens[2]*pec_cx[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)

    # plot all contributions
    if ax is None:
        fig,ax = plt.subplots()
        leg = True
    else:
        leg = False
    ax.plot(wave_final_A, spec_ion, c='r', label='ionization')
    ax.plot(wave_final_A, spec_exc, c='b', label='excitation')
    ax.plot(wave_final_A, spec_rec, c='g', label='radiative recomb')
    ax.plot(wave_final_A, spec_dr, c='m', label='dielectronic recomb')
    ax.plot(wave_final_A, spec_cx, c='c', label='charge exchange recomb')
    ax.plot(wave_final_A, spec_ion+spec_exc+spec_rec+spec_dr+spec_cx, c='k', label='total')
    if leg:
        # don't add legend and axis label if axes are passed as argument
        ax.legend(loc='best').set_draggable(True)
        ax.set_xlabel(r'$\lambda$ [$\AA$]')

    return ax



if __name__=='__main__':

    filepath_19='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca19.dat'
    filepath_18='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca18.dat'
    filepath_17='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca17.dat'
    
    Te_eV = 1000
    ne_cm3 = 1e14

    ax = plot_adas_spectrum(filepath_18, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
                       ion_exc_rec_dens=None)
    ax = plot_adas_spectrum(filepath_17, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
                            ion_exc_rec_dens=None, ax=ax)
    ax = plot_adas_spectrum(filepath_19, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
                       ion_exc_rec_dens=None)
