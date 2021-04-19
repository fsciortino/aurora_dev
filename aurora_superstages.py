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

# Use gfile and statefile in local directory:
examples_dir = '/home/sciortino/Aurora/examples'
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te']*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'Ca' #'Ar'
namelist['source_type'] = 'const'
namelist['Phi0'] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, polynomial V)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -1e2 * asim.rhop_grid**6 # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=False)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

# add radiation
#asim.rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te,
#                              prad_flag=True, thermal_cx_rad_flag=False, 
#                              spectral_brem_flag=False, sxr_flag=False)

# # plot radiation profiles over radius and time
# aurora.slider_plot(asim.rvol_grid, asim.time_out, asim.rad['line_rad'].transpose(1,2,0),
#                    xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
#                    labels=[str(i) for i in np.arange(0,nz.shape[1])],
#                    plot_sum=True, x_line=asim.rvol_lcfs)

# ----------------------
# Now test superstages
superstages = np.concatenate(([0,], np.arange(10,21)))
out2 = asim.run_aurora(D_z, V_z, superstages = superstages, plot=False)
nz2, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out2

aurora.slider_plot(asim.rvol_grid, asim.time_out, nz.transpose(1,0,2),
                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'nz [A.U.]',
                   labels=[str(i) for i in np.arange(0,nz.shape[1])],
                   plot_sum=True, x_line=asim.rvol_lcfs)

aurora.slider_plot(asim.rvol_grid, asim.time_out, nz2.transpose(1,0,2),
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
    a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
    if cs in superstages:
        a_plot.plot(asim.rhop_grid, nz2[:,scs,-1], ls, lw=3.)
        scs+=1
a_plot.set_xlabel(r'$\rho_p$')
a_plot.set_ylabel(r'$n_z$ [A.U.]')

a_legend.legend(loc='best', ncol=2).set_draggable(True)



####
#atom_data = aurora.atomic.get_atom_data(imp,['scd','acd'])

#logTe, fz, rates = aurora.atomic.get_frac_abundances(
#    atom_data, asim.ne, asim.Te, plot=True)


#supergroups = []
