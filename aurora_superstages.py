'''
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk, omfit_gapy
import sys, os
from scipy.interpolate import interp1d
import aurora
import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
#mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

device='ITER' #'CMOD' #'ITER'

if device=='ITER':
    case='/home/sciortino/ITER/ITER.baseline'
    profs = omfit_gapy.OMFITinputprofiles(case+'/input.profiles')
    geqdsk = omfit_eqdsk.OMFITgeqdsk('/home/sciortino/ITER/gfile_iter')
else:
    # CMOD
    # Use gfile and statefile in local directory:
    examples_dir = '/home/sciortino/Aurora/examples'
    geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
    profs = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(profs['polflux']/profs['polflux'][-1])
kp['ne']['vals'] = profs['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = profs['Te']*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'W' # 'Ca' #'W' #'Ca' #'Ar'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

namelist['timing'] = {'dt_increase': np.array([1.0, 1.   ]),
                      'dt_start': np.array([5.e-03, 1.e-03]),
                      'steps_per_cycle': np.array([1, 1]),
                      'times': np.array([0. , 1.5])}

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, polynomial V)
D_z = 5e2 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -10e3 * asim.rhop_grid**6 # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=False)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te,
                              prad_flag=True, thermal_cx_rad_flag=False, 
                              spectral_brem_flag=False, sxr_flag=False)
# plot radiation profiles over radius and time
aurora.slider_plot(asim.rvol_grid, asim.time_out, rad['line_rad'].transpose(1,2,0),
                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                   labels=[str(i) for i in np.arange(0,nz.shape[1])],
                   plot_sum=True, x_line=asim.rvol_lcfs)


# --------------------
# Now test superstages
if imp=='Ca':
    superstages = np.concatenate(([0,], np.arange(10,21)))
elif imp=='W':
    #superstages = np.concatenate((np.array([0,]), np.arange(50,70)))
    superstages = np.arange(0,75,2)
else:
    superstages = np.arange(0,nz.shape[1]-1,2)

out2 = asim.run_aurora(D_z, V_z, superstages = superstages, plot=False)
nz_super, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out2

aurora.slider_plot(asim.rvol_grid, asim.time_out, nz.transpose(1,0,2),
                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'nz [A.U.]',
                   labels=[str(i) for i in np.arange(0,nz.shape[1])],
                   plot_sum=True, x_line=asim.rvol_lcfs)

aurora.slider_plot(asim.rvol_grid, asim.time_out, nz_super.transpose(1,0,2),
                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'nz-super [A.U.]',
                   labels=[str(i) for i in superstages],
                   plot_sum=True, x_line=asim.rvol_lcfs)

# Only plot last time slice:
ls_cycle= aurora.get_ls_cycle()

fig = plt.figure()
fig.set_size_inches(12,7, forward=True)
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 7, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 3, fig=fig)
a_legend.axis('off')

scs=0
for cs in np.arange(nz.shape[1]):
    ls = next(ls_cycle) 
    a_plot.plot(asim.rhop_grid, nz[:,cs,-1],ls)
    if imp=='W' and device=='ITER':
        if cs>=superstages[1] and cs<=70:
            a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
    elif imp=='Ca' and device=='CMOD':
        if cs>10:
            a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
    else:
        a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
        
    if cs in superstages:
        a_plot.plot(asim.rhop_grid, nz_super[:,scs,-1], ls, lw=3.)
        scs+=1
a_plot.set_xlabel(r'$\rho_p$')
a_plot.set_ylabel(r'$n_z$ [A.U.]')

a_legend.legend(loc='best', ncol=1).set_draggable(True)


# ==========================================
####  Check approximation on radiation ####
# ==========================================

out3 = asim.run_aurora(D_z, V_z, superstages = superstages, unstage=True, plot=False)
nz3, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out3

#atom_data = aurora.atomic.get_atom_data(imp,['scd','acd'])

# get fractional abundances on ne (cm^-3) and Te (eV) grid
#logTe, fz = aurora.atomic.get_frac_abundances(
#    atom_data, asim.ne, asim.Te, plot=False)

# #nz_2 = np.zeros_like(fz)
# #for stage in np.arange(fz.shape[2]):
# #    for superstage in np.arange(nz_super.shape[1]):
# #        nz_2[...,stage] += nz_super[:,superstage,:].T * fz[..., stage]
# ## make of same shape as nz
# #nz_2 = nz_2.transpose(1,2,0)

# efficient sum over contributions from each superstage
#nz_2 = np.einsum('rst,trc->rct',nz_super,fz)

aurora.slider_plot(asim.rvol_grid, asim.time_out, nz3.transpose(1,0,2),
                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'nz [A.U.]',
                   labels=[str(i) for i in np.arange(0,nz3.shape[1])],
                   plot_sum=True, x_line=asim.rvol_lcfs)

rad_2= aurora.compute_rad(imp, nz3.transpose(2,1,0), asim.ne, asim.Te,
                              prad_flag=True, thermal_cx_rad_flag=False, 
                              spectral_brem_flag=False, sxr_flag=False)

# plot radiation profiles over radius and time
aurora.slider_plot(asim.rvol_grid, asim.time_out, rad_2['line_rad'].transpose(1,2,0),
                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                   labels=[str(i) for i in np.arange(0,nz3.shape[1])],
                   plot_sum=True, x_line=asim.rvol_lcfs)





# compare at last slice
ls_cycle= aurora.get_ls_cycle()

fig = plt.figure()
fig.set_size_inches(12,7, forward=True)
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 7, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 3, fig=fig)
a_legend.axis('off')

for cs in np.arange(nz.shape[1]):
    ls = next(ls_cycle) 
    a_plot.plot(asim.rhop_grid, nz[:,cs,-1],ls)
    a_plot.plot(asim.rhop_grid, nz3[:,cs,-1],ls, lw=3.)
    
    if imp=='W' and device=='ITER':
        if cs>=superstages[1] and cs<=70:
            a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
    elif imp=='Ca' and device=='CMOD':
        if cs>10:
            a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
    else:
        a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
        
    
a_plot.set_xlabel(r'$\rho_p$')
a_plot.set_ylabel(r'$n_z$ [A.U.]')

a_legend.legend(loc='best', ncol=1).set_draggable(True)
