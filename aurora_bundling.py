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


import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

ion =  'Ca' #'W' #'Ca' #'W'


Te_vals = np.linspace(10, 10000, 1000)
ne_vals = 5e13 * np.ones_like(Te_vals)

# get charge state distributions from ionization equilibrium for Ca
atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])

if ion=='W':
    fig,axs = plt.subplots(2,1, figsize=(12,8), sharex=True)
else:
    fig,axs = plt.subplots(figsize=(12,6), sharex=True)
    axs = np.atleast_1d(axs)

ax = axs[0] if ion=='Ca' else axs[1]

# get fractional abundances on ne (cm^-3) and Te (eV) grid
#Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne_vals, Te_vals, plot=False)

Te, S, R, cx = aurora.get_cs_balance_terms(atom_data, ne_vals, Te_vals, include_cx=False)

fig,ax = plt.subplots(figsize=(12,6), sharex=True)

# Using cumprod
rate_ratio = np.hstack((np.ones_like(Te)[:, None], S/R))
fz = np.cumprod(rate_ratio, axis=1)
fz /= fz.sum(1)[:, None]

# Plot fractional abundances in high resolution
x_fine = np.linspace(np.min(Te), np.max(Te),10000)
ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz.shape[1])))

for cs in range(fz.shape[1]):
    fz_i = interp1d(Te, fz[:,cs], kind='linear')(x_fine)
    ax.plot(x_fine, fz_i, ls='--')
ax.set_xscale('log')



# Using cumsum
fig,ax = plt.subplots(figsize=(12,6), sharex=True)
rate_ratio = np.hstack((np.zeros_like(np.log(Te))[:, None], np.log(S) - np.log(R)))
fz = np.exp(np.cumsum(rate_ratio, axis=1))
fz /= fz.sum(1)[:, None]

# Plot fractional abundances in high resolution
x_fine = np.linspace(np.min(Te), np.max(Te),10000)
ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz.shape[1])))

for cs in range(fz.shape[1]):
    fz_i = interp1d(Te, fz[:,cs], kind='linear')(x_fine)
    ax.plot(x_fine, fz_i, ls='-.')
ax.set_xscale('log')



# #####
# # Superstages / bundling
# Te_, S, R, cx = aurora.get_cs_balance_terms(atom_data, ne_vals, Te_vals, include_cx=False)
# rate_ratio = np.hstack((np.ones_like(Te_)[:, None], S/R))
# fz = np.cumprod(rate_ratio, axis=1)
# fz /= fz.sum(1)[:, None]


# # Different bundling method, choosing stages to be kept
# if ion=='W':
#     kept_stages = np.array([2*n**2 for n in np.arange(7)]) #np.concatenate((np.array([0.,]), np.arange(10,50,3)))
#     kept_stages = np.concatenate((kept_stages, np.array([74,])))

#     # choice where alpha_z+1~S_z is leveraged:
#     #kept_stages = np.concatenate((np.arange(31), np.arange(45,75)))
# else:
#     # Ca
#     kept_stages = np.array([0,10,11,12,13,14,15,16,17,18,19,20])


# fz_bundle = np.zeros((fz.shape[0], len(kept_stages)))
# ii=0
# for stage in np.arange(fz.shape[1]):
#     fz_bundle[:,ii] += fz[:,stage]
#     if stage in kept_stages:
#         ii+=1


# ax = axs[0] if ion=='Ca' else axs[1]
# x_fine = np.linspace(np.min(Te_), np.max(Te_),10000)
# ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz_bundle.shape[1])))

# for cs in range(fz_bundle.shape[1]):
#     fz_i = interp1d(Te_, fz_bundle[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz_i, ls='-.')

#     if ion=='W':
#         imax = np.argmax(fz_i)
#         ax.text(np.max([0.05,x_fine[imax]]), fz_i[imax], cs,
#                 horizontalalignment='center', clip_on=True)

# ########## Now superstage approximation #######
# # apply reduction to rates, rather than fractional abundances

# Te_, S, R, cx = aurora.get_cs_balance_terms(atom_data, ne_vals, Te_vals, include_cx=False)
# S_new = S[:,kept_stages[:-1]]  # no fully-stripped stage
# R_new = R[:,kept_stages[:-1]]  # no neutral stage
# rate_ratio = np.hstack((np.ones_like(Te_)[:, None], S_new/R_new))
# fz_super = np.cumprod(rate_ratio, axis=1)
# fz_super /= fz_super.sum(1)[:, None]



# ax = axs[0] if ion=='Ca' else axs[1]
# x_fine = np.linspace(np.min(Te_), np.max(Te_),10000)
# ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz_super.shape[1])))

# for cs in range(fz_super.shape[1]):
#     fz_i = interp1d(Te_, fz_super[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz_i, ls='-', lw=3.)


# # ========================

# # breakdown of superstages into original stages using ionization equilibrium
# fz_back = np.zeros_like(fz)
# fz_back2 = np.zeros_like(fz)

# for stage in np.arange(fz.shape[1]):
#     for superstage in np.arange(fz_super.shape[1]):
#         fz_back[:,stage] += fz_super[:,superstage]*fz[:,stage]
#         fz_back2[:,stage] += fz_bundle[:,superstage]*fz[:,stage]
        

# ax = axs[0]
# for cs in range(fz.shape[1]):
#     fz_i = interp1d(Te_, fz_back[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz_i, ls='-') #, lw=3.)




# plt.tight_layout()
# axs[-1].set_xlabel('T$_e$ [eV]')
# fig.subplots_adjust(hspace=0.0)


# fig,ax = plt.subplots()
# for cs in range(fz.shape[1]):
#     fz_i = interp1d(Te_, fz_back[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz_i, ls='-') #, lw=3.)

#     fz_i = interp1d(Te_, fz_back2[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz_i, ls='--') #, lw=3.)




# fig,ax = plt.subplots()
# for cs in range(fz_super.shape[1]):
#     fz_i = interp1d(Te_, fz_bundle[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz_i, ls='-') #, lw=3.)

#     fz_i = interp1d(Te_, fz_super[:,cs], kind='cubic')(x_fine)
#     ax.plot(x_fine, fz
#            _i, ls='--') #, lw=3.)
