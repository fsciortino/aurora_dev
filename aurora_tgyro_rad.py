'''Cooling curves for multiple ions from Aurora. 

No charge exchange included. Simple ionization equilibrium.

sciortino, 2021
'''

import aurora
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# scan Te and fix a value of ne
Te_eV = np.logspace(np.log10(100), np.log10(1e5), 1000)
ne_cm3 = 5e13 * np.ones_like(Te_eV)
#pue_base = '/fusion/projects/toolbox/sciortinof/atomlib/atomdat_master/pue2020_data/'
pue_base =  '/home/sciortino/atomlib/atomdat_master/pue2020_data/'

fig1,ax1 = plt.subplots()

ions_list = ['He','C','O','Ar','W'] #['H','He','Be','B','C','O','N','F','Ne','Al','Si','Ar','Ca','Fe','Ni','Kr','Mo','Xe','W']


for imp in ions_list:

    # read atomic data, interpolate and plot cooling factors
    line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
        imp, ne_cm3, Te_eV, show_components=False, ax=ax1,
        #line_rad_file=pue_base+f'plt_ca_{imp}.dat',
        #cont_rad_file=pue_base+f'prb_{imp}.dat'
        )

    # read atomic data, interpolate and plot cooling factors
    line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
        imp, ne_cm3, Te_eV, show_components=False, ax=ax1,
        line_rad_file=pue_base+f'plt_ca_{imp}.dat',
        cont_rad_file=pue_base+f'prb_{imp}.dat'
        )

######
# Can use the following for other ions:
# for imp in ['O','N','Ne']:
    
#     # read atomic data, interpolate and plot cooling factors (use Aurora defaults, not Puetterich data)
#     line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
#         imp, ne_cm3, Te_eV, show_components=False, ax=ax1)

# # NB: total cooling curve is simply cont_rad_tot+line_rad_tot

    
