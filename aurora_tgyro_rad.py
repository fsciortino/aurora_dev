'''Cooling curves for multiple ions from Aurora. 

No charge exchange included. Simple ionization equilibrium.

sciortino, 2021
'''

import aurora
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16


# scan Te and fix a value of ne
Te_eV = np.logspace(np.log10(100), np.log10(1e5), 1000)
ne_cm3 = 5e13 * np.ones_like(Te_eV)
pue_base = '/fusion/projects/toolbox/sciortinof/atomlib/atomdat_master/pue2021_data/'

fig = plt.figure(figsize=(10,7))
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 8, fig=fig) 

a_legend.axis('off')



#ions_list = ['He','C','Ar','W'] 
#ions_list = ['H','He','Be','B','C','O','N','F','Ne','Al','Si','Ar','Ca','Fe','Ni','Kr','Mo','Xe','W']
ions_list = ['H','He','Be','C','N','F','Ne','Al','Ar','Ca','Fe','Kr','Mo','Xe','W']


ls_cycle = aurora.get_ls_cycle()
for imp in ions_list:

    ls = next(ls_cycle)

    # read atomic data, interpolate and plot cooling factors
    line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
        imp, ne_cm3, Te_eV, show_components=False, plot=False, #ax=ax1,
        line_rad_file=pue_base+f'plt_caic_mix_{imp}.dat',
        cont_rad_file=pue_base+f'prb_{imp}.dat'
        )

    atom_data = aurora.get_atom_data(imp,{'plt': pue_base+f'plt_caic_mix_{imp}.dat'})
    #print(f'Temperature range of validity for {imp}: [{10**np.min(atom_data["plt"][1]):.1f}, {10**np.max(atom_data["plt"][1]):.1f}] eV')
    
    # total radiation (includes hard X-ray, visible, UV, etc.)
    a_plot.loglog(Te_eV/1e3, cont_rad_tot+line_rad_tot, ls)
    a_legend.plot([],[], ls, label=f'{imp}')
    
    
a_legend.legend(loc='best').set_draggable(True)
a_plot.grid('on', which='both')
a_plot.set_xlabel('T$_e$ [keV]')
a_plot.set_ylabel('$L_z$ [$W$ $m^3$]')
plt.tight_layout()

######
# Can use the following for other ions:
# for imp in ['O','N','Ne']:
    
#     # read atomic data, interpolate and plot cooling factors (use Aurora defaults, not Puetterich data)
#     line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
#         imp, ne_cm3, Te_eV, show_components=False, ax=ax1)

# # NB: total cooling curve is simply cont_rad_tot+line_rad_tot

    
