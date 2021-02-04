import numpy as np, sys
import matplotlib.pyplot as plt
from scipy.constants import e as q_electron,k as k_B, h, m_p, c as c_speed
plt.ion()
import periodictable
from scipy.interpolate import interp1d
import aurora

from IPython import embed

def plot_adas_spectrum(adf15_filepath, ion, ne_cm3, Te_eV, n0_cm3=0.0,
                       ion_exc_rec_dens=None, ax=None, plot_spec_tot=True, no_leg=False):
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
    plot_spec_tot : bool
        If True, plot total spectrum (sum over all components) from given ADF15 file. 
    no_leg : bool
        If True, no plot legend is shown. Default is False, i.e. show legend.

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
    
    # Doppler broadening
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

    spec_tot = spec_ion+spec_exc+spec_rec+spec_dr+spec_cx
    
    # plot all contributions
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(wave_final_A, spec_ion, c='r', label='' if no_leg else 'ionization')
    ax.plot(wave_final_A, spec_exc, c='b', label='' if no_leg else 'excitation')
    ax.plot(wave_final_A, spec_rec, c='g', label='' if no_leg else 'radiative recomb')
    ax.plot(wave_final_A, spec_dr, c='m', label='' if no_leg else 'dielectronic recomb')
    ax.plot(wave_final_A, spec_cx, c='c', label='' if no_leg else 'charge exchange recomb')
    if plot_spec_tot:
        ax.plot(wave_final_A, spec_tot, c='k', label='' if no_leg else 'total')
        
    if no_leg:
        ax.legend(loc='best').set_draggable(True)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r'$\epsilon$ [A.U.]')

    return wave_final_A, spec_tot, ax



if __name__=='__main__':

    filepath_19='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca19.dat'
    filepath_18='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca18.dat'
    filepath_17='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca17.dat'
    
    Te_eV = 1e3 #500
    ne_cm3 = 1e14

    fig = plt.figure()
    fig.set_size_inches(10,7, forward=True)
    ax1 = plt.subplot2grid((10,1),(0,0),rowspan = 1, colspan = 1, fig=fig)
    ax2 = plt.subplot2grid((10,1),(1,0),rowspan = 9, colspan = 1, fig=fig, sharex=ax1)

    ax2.set_xlim([3.17, 3.215]) # A, He-lke Ca spectrum
    
    with open('/home/sciortino/usr/python3modules/bsfc/data/hirexsr_wavelengths.csv', 'r') as f:
        lineData = [s.strip().split(',') for s in f.readlines()]
        lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
        lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
        lineName = np.array([ld[3] for ld in lineData[2:]])
        
    for ii,_line in enumerate(lineLam):
        if _line>ax2.get_xlim()[0] and _line<ax2.get_xlim()[1]:
            ax2.axvline(_line, c='r', ls='--')
            ax1.axvline(_line, c='r', ls='--')
            
            ax1.text(_line, 0.5, lineName[ii], rotation=90, fontdict={'fontsize':14}) #, transform=ax1.transAxes)
    ax1.axis('off')

    # now add spectra
    wave_final_A_18, spec_tot_18, ax = plot_adas_spectrum(
        filepath_18, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
        ion_exc_rec_dens=None, ax=ax2,  plot_spec_tot=False)
    wave_final_A_17, spec_tot_17, ax = plot_adas_spectrum(
        filepath_17, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
        ion_exc_rec_dens=None, ax=ax2, plot_spec_tot=False, no_leg=True)
    wave_final_A_19, spec_tot_19, ax = plot_adas_spectrum(
        filepath_19, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
        ion_exc_rec_dens=None, ax=ax2, plot_spec_tot=False, no_leg=True)

    # add plot of total spectrum
    wave_all_A = np.linspace(3.17,3.24, 10000)
    spec_all = interp1d(wave_final_A_18, spec_tot_18, bounds_error=False, fill_value=0.0)(wave_all_A)
    spec_all += interp1d(wave_final_A_17, spec_tot_17, bounds_error=False, fill_value=0.0)(wave_all_A)
    spec_all += interp1d(wave_final_A_19, spec_tot_19, bounds_error=False, fill_value=0.0)(wave_all_A)
    plt.gca().plot(wave_all_A, spec_all, 'k', label='total')
    plt.gca().legend(loc='best').set_draggable(True)
