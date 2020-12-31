import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import sys
from scipy.interpolate import interp1d
import aurora

sys.path.insert(1, '/home/sciortino/atomID/coreSPRED')
import get_zipfit
import coreSPRED_helper
from scipy.interpolate import interpn

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()
kp = namelist['kin_profs']

# shot and time
shot=180526
time=2750  #ms

# get ZIPFIT kinetic profiles
zipfit = get_zipfit.load_zipfit(shot)

# get ne,Te profiles at chosen time
tidx = np.argmin(np.abs(zipfit['ne']['tvec'].data - time/1e3))
kp['ne']['vals'] = ne = zipfit['ne']['data'].data[tidx,:]*1e-6 # cm^-3
kp['Te']['vals'] = Te = zipfit['Te']['data'].data[tidx,:]   # eV
rho_tor = zipfit['ne']['rho'].data

# get coreSPRED line integration path
pathR, pathZ, pathL = coreSPRED_helper.get_spred_geom()
