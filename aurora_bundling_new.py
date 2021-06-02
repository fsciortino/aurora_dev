import copy
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_gapy
import scipy,sys,os
import time
from scipy.interpolate import interp1d
from matplotlib import cm
import aurora

#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)

import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16


Te_vals = np.geomspace(1e1,1e4, 1000) #np.linspace(10, 10000, 1000)
ne_vals = 5e13 * np.ones_like(Te_vals)


fig, axs = plt.subplots(2,1, figsize=(10,8))

####
ion = 'W' 
atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])
superstages = np.array([0,2,8,18,32,50,72,74])
logTe, fz = aurora.atomic.get_frac_abundances(
    atom_data, ne_vals, Te_vals, plot=True, ax=axs[0], superstages=superstages)



######
ion = 'Ca'
atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])
superstages = np.array([0,10,11,12,13,14,15,16,17,18,19,20])

logTe, fz = aurora.atomic.get_frac_abundances(
    atom_data, ne_vals, Te_vals, plot=True, ax=axs[1], superstages=superstages)

fig.subplots_adjust(hspace=0.0)

plt.gcf().get_axes()[0].set_xticklabels([])
