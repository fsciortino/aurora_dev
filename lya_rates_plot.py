'''This script is a small modification of read_ADF15 in Aurora, for the purpose of plotting Ly-a rates nicely.

sciortino,2021
'''


import aurora
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from IPython import embed
from scipy.interpolate import interp1d, RectBivariateSpline

import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16


def read_adf15(path, order=1, plot_lines=[], ax=None, plot_3d=False):
    """Read photon emissivity coefficients from an ADAS ADF15 file.

    Returns a dictionary whose keys are the wavelengths of the lines in angstroms. 
    The value is an interpolant that will evaluate the log10 of the PEC at a desired density 
    and temperature. The power-10 exponentiation of this PEC has units of :math:`photons \cdot cm^3/s`

    Units for interpolation: :math:`cm^{-3}` for density; :math:`eV` for temperature.

    Parameters
    ----------
    path : str
        Path to adf15 file to read.
    order : int, opt
        Parameter to control the order of interpolation. Default is 1 (linear interpolation).
    plot_lines : list
        List of lines whose PEC data should be displayed. Lines should be identified
        by their wavelengths. The list of available wavelengths in a given file can be retrieved
        by first running this function ones, checking dictionary keys, and then requesting a
        plot of one (or more) of them.
    ax : matplotlib axes instance
        If not None, plot on this set of axes.
    plot_3d : bool
        Display PEC data as 3D plots rather than 2D ones.

    Returns
    -------
    log10pec_dict : dict
        Dictionary containing interpolation functions for each of the available lines of the
        indicated type (ionization or recombination). Each interpolation function takes as arguments
        the log-10 of ne and Te and returns the log-10 of the chosen PEC.
    
    Examples
    --------
    To plot the Lyman-alpha photon emissivity coefficients for H (or its isotopes), you can use:

    >>> filename = 'pec96#h_pju#h0.dat' # for D Ly-alpha
    >>> # fetch file automatically, locally, from AURORA_ADAS_DIR, or directly from the web:
    >>> path = aurora.get_adas_file_loc(filename, filetype='adf15')  
    >>>
    >>> # plot Lyman-alpha line at 1215.2 A. 
    >>> # see available lines with log10pec_dict.keys() after calling without plot_lines argument
    >>> log10pec_dict = aurora.read_adf15(path, plot_lines=[1215.2])

    Another example, this time also with charge exchange::

    >>> filename = 'pec96#c_pju#c2.dat'
    >>> path = aurora.get_adas_file_loc(filename, filetype='adf15')
    >>> log10pec_dict = aurora.read_adf15(path, plot_lines=[361.7])

    Metastable-resolved files will be automatically identified and parsed accordingly, e.g.::

    >>> filename = 'pec96#he_pjr#he0.dat'
    >>> path = aurora.get_adas_file_loc(filename, filetype='adf15')
    >>> log10pec_dict = aurora.read_adf15(path, plot_lines=[584.4])

    Notes
    -----
    This function expects the format of PEC files produced via the ADAS adas810 or adas218 routines.

    """
    # find out whether file is metastable resolved
    meta_resolved = path.split('#')[-2][-1]=='r'
    if meta_resolved: print('Identified metastable-resolved PEC file')
    
    with open(path, 'r') as f:
        lines = f.readlines()
    cs = path.split('#')[-1].split('.dat')[0]

    header = lines.pop(0)
    # Get the expected number of lines by reading the header:
    num_lines = int(header.split()[0])
    log10pec_dict = {}

    for i in range(0, num_lines):
        
        if '----' in lines[0]: 
            _ = lines.pop(0) # separator may exist before each transition

        # Get the wavelength, number of densities and number of temperatures
        # from the first line of the entry:
        l = lines.pop(0)
        header = l.split()

        # sometimes the wavelength and its units are not separated:
        try:
            header = [hh.split('A')[0] for hh in header]
        except:
            # lam and A are separated. Delete 'A' unit.
            header = np.delete(header, 1)

        lam = float(header[0])

        if header[1]=='':
            # 2nd element was empty -- annoyingly, this happens sometimes
            num_den = int(header[2])
            num_temp = int(header[3])
        else:
            num_den = int(header[1])
            num_temp = int(header[2])            

        if meta_resolved:
            # index of metastable state
            INDM = int(header[-3].split('/')[0].split('=')[-1])

        # Get the densities:
        dens = []
        while len(dens) < num_den:
            dens += [float(v) for v in lines.pop(0).split()]
        dens = np.asarray(dens)

        # Get the temperatures:
        temp = []
        while len(temp) < num_temp:
            temp += [float(v) for v in lines.pop(0).split()]
        temp = np.asarray(temp)

        # Get the PEC's:
        PEC = []
        while len(PEC) < num_den:
            PEC.append([])
            while len(PEC[-1]) < num_temp:
                PEC[-1] += [float(v) for v in lines.pop(0).split()]
        PEC = np.asarray(PEC)
        
        # find what kind of rate we are dealing with
        if 'recom' in l.lower(): rate_type = 'recom'
        elif 'excit' in l.lower(): rate_type = 'excit'
        elif 'chexc' in l.lower(): rate_type = 'chexc'
        elif 'drsat' in l.lower(): rate_type = 'drsat'
        elif 'ion' in l.lower(): rate_type = 'ioniz'
        else:
            # attempt to report unknown rate type -- this should be fairly robust
            rate_type = l.replace(' ','').lower().split('type=')[1].split('/')[0]

        # create dictionary with keys for each wavelength:
        if lam not in log10pec_dict:
            log10pec_dict[lam] = {}                

        # add a key to the log10pec_dict[lam] dictionary for each type of rate: recom, excit or chexc
        # interpolate PEC on log dens,temp scales
        pec_fun = RectBivariateSpline(
            np.log10(dens),
            np.log10(temp),
            np.log10(PEC),   # NB: interpolation of log10 of PEC to avoid issues at low ne or Te
            kx=order,
            ky=order
        )
        
        if meta_resolved:
            if rate_type not in log10pec_dict[lam]:
                log10pec_dict[lam][rate_type] = {}
            log10pec_dict[lam][rate_type][f'meta{INDM}'] = pec_fun
        else:
            log10pec_dict[lam][rate_type] = pec_fun
            
        if lam in plot_lines:

            # only plot 3 densities at chosen indices
            dens_idx = np.array([13, 15, 16,18, 19])

            # plot PEC values over ne,Te grid given by ADAS, showing interpolation quality
            NE, TE = np.meshgrid(dens[dens_idx], temp)
            
            PEC_eval = 10**pec_fun.ev(np.log10(NE), np.log10(TE)).T

            # plot PEC rates
            _ax = _plot_pec(dens[dens_idx],temp, PEC[dens_idx,:], PEC_eval, lam,cs,rate_type, ax, plot_3d)

            meta_str = ''
            if meta_resolved: meta_str = f' , meta = {INDM}'
            #_ax.set_title(cs + r' , $\lambda$ = '+str(lam) +' $\AA$, '+rate_type+meta_str)
            plt.tight_layout()

    return log10pec_dict



def _plot_pec(dens, temp, PEC, PEC_eval, lam,cs,rate_type, ax=None, plot_3d=False):
    '''Private method to plot PEC data within :py:func:`~aurora.atomic.read_adf15` function.
    '''
    if ax is None:
        f1 = plt.figure() #figsize=(7,6))
        ax1 = f1.add_subplot(1,1,1)
    else:
        ax1 = ax

    # plot in 2D
    labels = ['{:.0e}'.format(ne)+r' $cm^{-3}$' for ne in dens] #ne_eval]

    for ine in np.arange(PEC.shape[0]):
        l, = ax1.plot(temp, PEC_eval[ine,:], label=labels[ine])
        ax1.plot(temp, PEC[ine,:], color=l.get_color(), marker='o', mfc=l.get_color(), ms=5.)

    ax1.set_xlabel(r'$T_e$ [eV]')
    ax1.set_ylabel('PEC [photons $\cdot cm^3/s$]')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim([10,1e4])

    #ax1.set_ylim([6e-9,6e-8])
    ax1.legend(loc='best').set_draggable(True)
    ax1.grid('on', which='both')
    
    return ax1


filename = 'pec96#h_pju#h0.dat' # for D Ly-alpha

# fetch file automatically, locally, from AURORA_ADAS_DIR, or directly from the web:
path = aurora.get_adas_file_loc(filename, filetype='adf15')


log10pec_dict = read_adf15(path, plot_lines=[1215.2])


###### Now fetch ionization and recombination rates
atom_data = aurora.atomic.get_atom_data('H',['scd','acd'])

#ne_prof = np.array([1e12,5e12,1e13,5e13,1e14])


fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()

for ne in [1e12,5e12,1e13,5e13,1e14]:
    Te_prof = np.linspace(10., 1e4, 1000)
    ne_prof = ne*np.ones_like(Te_prof)
    lne = np.log10(ne_prof)
    lTe = np.log10(Te_prof)
    
    S_rates = aurora.interp_atom_prof(atom_data['scd'],lne, lTe)
    R_rates = aurora.interp_atom_prof(atom_data['acd'],lne, lTe)
    
    ax1.loglog(Te_prof, S_rates[:,0])
    ax1.set_xlabel(r'$T_e$ [eV]')
    ax1.set_ylabel(r'$S$ [s$^{-1}$]')

    ax2.loglog(Te_prof, R_rates[:,0], label=fr'$n_e={ne:.0g}$ cm$^{{{-3}}}$')
    ax2.set_xlabel(r'$T_e$ [eV]')
    ax2.set_ylabel(r'$R$ [s$^{-1}$]')

ax2.legend(loc='best').set_draggable(True)
ax1.grid('on', which='both')
ax1.set_xlim([10,1e4])
plt.tight_layout()
ax2.grid('on', which='both')
plt.tight_layout()
ax2.set_xlim([10,1e4])
