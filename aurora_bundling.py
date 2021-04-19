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


ion = 'Ca' #'W'

Te_vals = np.linspace(10, 10000, 1000)
ne_vals = 5e13 * np.ones_like(Te_vals)

# get charge state distributions from ionization equilibrium for Ca
atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])

# get fractional abundances on ne (cm^-3) and Te (eV) grid
logTe, fz, rates = aurora.atomic.get_frac_abundances(
    atom_data, ne_vals, Te_vals, plot=True)

#####
# Superstages / bundling
logTe_, logS, logR, logcx = aurora.get_cs_balance_terms(atom_data, ne_vals, Te_vals, maxTe=10e3, include_cx=False)

# analytical formula for fractional abundance
rate_ratio = np.hstack((np.zeros_like(logTe_)[:, None], logS - logR))
fz = np.exp(np.cumsum(rate_ratio, axis=1))
fz /= fz.sum(1)[:, None]

def reduce_array(arr, idx, bundle_up=True):
    tobundle = arr[:,idx]
    arr_new = np.delete(arr, idx, axis=1)
    arr_new[:,idx if bundle_up else idx-1] += tobundle
    
    return arr_new

# bundle selected stages
#bundled_stages = np.concatenate((np.arange(1,30), np.arange(50,fz.shape[1])))
# bundled_stages = np.arange(1,fz.shape[1]-1)[::2]
# fz_new = copy.deepcopy(fz)
# for stage in bundled_stages[::-1]:
#     fz_new = reduce_array(fz_new, stage, bundle_up=True)
    
# ax = plt.gca()
# x_fine = np.linspace(np.min(10**logTe_), np.max(10**logTe_),10000)
# ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz_new.shape[1])))

# for cs in range(fz_new.shape[1]):
#     fz_i = interp1d(10**logTe_, fz_new[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz_i, ls='--')

# Different bundling method, choosing stages to be kept
#kept_stages = np.arange(0,30) #[0,10,20,30,40,50,60,73]
kept_stages = np.array([0,10,11,12,13,14,15,16,17,18,19,20])

fz_new2 = copy.deepcopy(fz)
for stage in np.arange(fz.shape[1]-1)[::-1]:
    if stage not in kept_stages:
        fz_new2 = reduce_array(fz_new2, stage, bundle_up=True)

ax = plt.gca()
x_fine = np.linspace(np.min(10**logTe_), np.max(10**logTe_),10000)
ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz_new2.shape[1])))

for cs in range(fz_new2.shape[1]):
    fz_i = interp1d(10**logTe_, fz_new2[:,cs], kind='cubic')(x_fine)
    ax.plot(x_fine, fz_i, ls='-.')


##########
logTe_, logS, logR, logcx = aurora.get_cs_balance_terms(atom_data, ne_vals, Te_vals, maxTe=10e3, include_cx=False)

# apply reduction to rates, rather than fractional abundances
logS_new = copy.deepcopy(logS)
logR_new = copy.deepcopy(logR)
for stage in np.arange(logS.shape[1]-1)[::-1]:
    if stage not in kept_stages:
        logS_new = np.delete(logS_new, stage, axis=1)
        logR_new = np.delete(logR_new, stage, axis=1)

# analytical formula for fractional abundance
rate_ratio = np.hstack((np.zeros_like(logTe_)[:, None], logS_new - logR_new))
fz3 = np.exp(np.cumsum(rate_ratio, axis=1))
fz3 /= fz3.sum(1)[:, None]

ax = plt.gca()
x_fine = np.linspace(np.min(10**logTe_), np.max(10**logTe_),10000)
ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz3.shape[1])))

for cs in range(fz3.shape[1]):
    fz_i = interp1d(10**logTe_, fz3[:,cs], kind='cubic')(x_fine)
    ax.plot(x_fine, fz_i, ls='-', lw=3.)

