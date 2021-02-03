'''Automated plots of total radiation density from Aurora for a number of ions.
Results in W/cm^3.

No charge exchange included. Simple ionization equilibrium.
'''

import aurora
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

plt.style.use('/home/sciortino/SPARC/sparc_plots.mplstyle')


def get_cooling_factors(imp, ne_cm3, Te_eV, plot=True, show_components=True, ax=None):
    '''Calculate cooling coefficients for the given fractional abundances and kinetic profiles.

    Parameters
    ----------

    plot : bool
        If True, plot all radiation components, summed over charge states.
    ax : matplotlib.Axes instance
        If provided, plot results on these axes. 
    
    Returns
    -------

    '''
    atom_data = aurora.atomic.get_atom_data(imp,['scd','acd'])
    logTe, fz, rates = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, plot=False)

    atom_data = aurora.atomic.get_atom_data(imp,['plt','prb'])
    pltt= aurora.interp_atom_prof(atom_data['plt'],None, np.log10(Te_eV)) # line radiation [W.cm^3]
    prb = aurora.interp_atom_prof(atom_data['prb'],None, np.log10(Te_eV)) # continuum radiation [W.cm^3]

    pltt*= fz[:,:-1]
    prb *= fz[:, 1:]

    line_rad_tot  = pltt.sum(1) *1e-6  # W.cm^3-->W.m^3
    cont_rad_tot = prb.sum(1) *1e-6    # W.cm^3-->W.m^3

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        # total radiation (includes hard X-ray, visible, UV, etc.)
        l, = ax.loglog(Te_eV/1e3, cont_rad_tot+line_rad_tot, ls='-', label=f'{imp} $L_z$ (total)' if show_components else f'{imp}')
        col = l.get_color()
        
        if show_components:
            ax.loglog(Te_eV/1e3, line_rad_tot,c=col, ls='--',label='line radiation')
            ax.loglog(Te_eV/1e3, cont_rad_tot,c=col, ls='-.',label='continuum radiation')
    
        ax.legend(loc='best')

        ax.grid('on')
        ax.set_xlabel('T$_e$ [keV]')
        ax.set_ylabel('Cooling factor $L_z$ [$W$ $m^3$]')
        plt.tight_layout()
        
    # ion-resolved radiation terms:
    return line_rad_tot, cont_rad_tot



fig1,ax1 = plt.subplots()

for imp in ['He','N','Ne','Ar','W']:

    Te_eV = np.logspace(np.log10(10), np.log10(1e5), 1000)
    ne_cm3 = 5e13 * np.ones_like(Te_val)

    out = get_cooling_factors(imp, ne_cm3, Te_eV, ax=ax1)
