'''Cooling curves for multiple ions from Aurora. 

No charge exchange included. Simple ionization equilibrium.

sciortino, 2021
'''

import aurora
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

fig1,ax1 = plt.subplots()

for imp in ['He','C','O','Ar','W']:

    # scan Te and fix a value of ne
    Te_eV = np.logspace(np.log10(10), np.log10(1e5), 1000)
    ne_cm3 = 5e13 * np.ones_like(Te_eV)
    
    # read atomic data, interpolate and plot cooling factors
    line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
        imp, ne_cm3, Te_eV, show_components=False, ax=ax1
        )

    # NB: total cooling curve is simply cont_rad_tot+line_rad_tot
