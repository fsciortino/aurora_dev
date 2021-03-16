'''Aurora-SOLPS coupling methods.

sciortino, 2021
'''
import pickle as pkl
import matplotlib.pyplot as plt
import MDSplus, os, copy, sys
import numpy as np
plt.ion()
from scipy.interpolate import interp1d
import aurora
from heapq import nsmallest
from IPython import embed
#plt.style.use('/home/sciortino/SPARC/sparc_plots.mplstyle')
from scipy import constants

import matplotlib.colors as colorsMPL
import math
import os.path

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

if '/home/sciortino/tools3/' not in sys.path:
    sys.path.append('/home/sciortino/tools3/')

from plot_cmod_machine import overplot_machine    

import aurora


import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16


# L-mode (J.Rice)
shot = 1120917011; solps_run='Attempt75' 
path = f'/home/sciortino/SOLPS/full_CMOD_runs/Lmode_1120917011/'
gfilepath = f'/home/sciortino/EFIT/gfiles/g{shot}.00999_981'  # hard-coded

# load SOLPS results
so = aurora.solps_case(path, gfilepath, solps_run=solps_run,form='full')


# plot some important fields
fig,ax = plt.subplots(figsize=(8,10))
overplot_machine(shot, ax)
so.plot2d_b2(so.quants['nn'], ax=ax, scale='log', label=so.labels['nn'])
so.geqdsk.plot(only2D=True, ax=ax)


# first get radial slice
rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS = so.get_radial_prof(so.quants['nn'], plot=True)


# now vertical slice

from scipy.interpolate import griddata

Z = np.linspace(np.min(so.Z), np.max(so.Z), 200)
R = 0.69 * np.ones_like(Z)
nn_vert = griddata((so.R.flatten(),so.Z.flatten()), so.quants['nn'].flatten(),
         (R,Z), method='linear')

mask = ~np.isnan(nn_vert)
Z = Z[mask]
nn_vert = nn_vert[mask]

plt.figure()
plt.semilogy(Z, nn_vert)
plt.xlabel('Z [m]')
plt.ylabel(r'$n_n$ $[m^{-3}]$')
plt.tight_layout()


with open(f'nn_{shot}_jr_cxr.dat','w+') as f:
    f.write('Midplane atomic neutral D ground state densities [m^-3] \n')
    f.write(' \n')
    f.write(f'HFS rhop ({len(rhop_HFS)} values) \n')
    [f.write(f'{val:.4f} ') for val in rhop_HFS]
    f.write(' \n')
    f.write(' \n')
    f.write(f'HFS nn ({len(neut_HFS)} values) \n')
    [f.write(f'{val:.4f} ') for val in neut_HFS]
    f.write(' \n')
    f.write(' \n')
    f.write(f'LFS rhop ({len(rhop_LFS)} values) \n')
    [f.write(f'{val:.4f} ') for val in rhop_LFS]
    f.write(' \n')
    f.write(' \n')
    f.write(f'LFS nn ({len(neut_LFS)} values) \n')
    [f.write(f'{val:.4f} ') for val in neut_LFS]
    f.write(' \n')
    f.write(' \n')
    f.write('Vertical slice at R=0.69m \n')
    f.write('Z [m] ({len(Z)} values) \n')
    [f.write(f'{val:.4f} ') for val in Z]
    f.write(' \n')
    f.write(' \n')
    [f.write(f'{val:.4f} ') for val in nn_vert]

    f.write(' \n')
    f.write(' \n')
