'''Automated plots of total radiation density from Aurora for a number of ions.
Results in W/cm^3.

No charge exchange included. Simple ionization equilibrium.
'''

import aurora
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

plt.style.use('/home/sciortino/SPARC/sparc_plots.mplstyle')



fig1,ax1 = plt.subplots()

for imp in ['He','N','Ne','Ar','W']:

    Te_eV = np.logspace(np.log10(10), np.log10(1e5), 1000)
    ne_cm3 = 5e13 * np.ones_like(Te_eV)

    out = aurora.get_cooling_factors(imp, ne_cm3, Te_eV, show_components=False, ax=ax1)
