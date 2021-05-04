'''
Test of reading of Puetterich 2020 atomic data
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import sys, os
from scipy.interpolate import interp1d
import aurora

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
imp = namelist['imp'] = 'Ca'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# check radial grid:
_ = aurora.create_radial_grid(namelist,plot=True)

# check time grid:
_ = aurora.create_time_grid(namelist['timing'], plot=True)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

# add radiation
asim.rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te,
                              prad_flag=True, thermal_cx_rad_flag=False, 
                              spectral_brem_flag=False,
                              sxr_flag=True,
                              adas_files_sub={'pls':'/home/sciortino/atomlib/atomdat_master/pue2020_data/pls_Ca_9.dat',
                                              'prs':'/home/sciortino/atomlib/atomdat_master/pue2020_data/prs_Ca_9.dat'})

# plot radiation profiles over radius and time
aurora.slider_plot(asim.rhop_grid, asim.time_out,
                   asim.rad['sxr_line_rad'].transpose(1,2,0),
                   xlabel=r'$\rho_p$', ylabel='time [s]',
                   zlabel=r'SXR Line radiation [$MW/m^3$]',
                   labels=[str(i) for i in np.arange(0,nz.shape[1])],
                   plot_sum=True, x_line=1.0)
