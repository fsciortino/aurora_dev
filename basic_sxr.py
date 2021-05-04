'''
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import sys
from scipy.interpolate import interp1d
import aurora

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
geqdsk = omfit_eqdsk.OMFITgeqdsk('/home/sciortino/Aurora/examples/example.gfile')
inputgacode = omfit_gapy.OMFITgacode('/home/sciortino/Aurora/examples/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te']*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'Ar'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

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
                              spectral_brem_flag=False, sxr_flag=True)

# plot radiation profiles over radius and time
aurora.slider_plot(asim.rvol_grid, asim.time_out, asim.rad['sxr_line_rad'].transpose(1,2,0),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)


###############################

imp = namelist['imp'] = 'F'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

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
                              spectral_brem_flag=False, sxr_flag=True)

# plot radiation profiles over radius and time
aurora.slider_plot(asim.rvol_grid, asim.time_out, asim.rad['sxr_line_rad'].transpose(1,2,0),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)

