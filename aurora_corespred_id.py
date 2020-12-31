'''Attempt at identifying atomic lines using Aurora charge state time traces. 

sciortino, 2020
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import sys
from scipy.interpolate import interp1d
import aurora
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

geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
                device='DIII-D',shot=shot,time=time, SNAPfile='EFIT01'
                )

psiRZ = geqdsk['AuxQuantities']['PSIRZ_NORM'].T
eq_R = geqdsk['AuxQuantities']['R']
eq_Z = geqdsk['AuxQuantities']['Z']

pathPsi = interpn((eq_R, eq_Z), psiRZ, np.c_[pathR, pathZ], bounds_error=False, fill_value=2)

# get line integration weights
kp['ne']['rhop'] = kp['Te']['rhop'] = rhop = aurora.coords.rad_coord_transform(rho_tor, 'rhon','rhop', geqdsk)
rhop_lineint = np.linspace(np.min(rhop),np.max(rhop),200)
weights_lineint = aurora.line_int_weights(pathR,pathZ,np.sqrt(pathPsi),pathL,rhop_out=rhop_lineint)

# don't use any PEC dependencies for simplicity
exc_pec_vals = 1e-26   # fixed here


# set impurity species and sources rate
imp = namelist['imp'] = 'F'
namelist['source_type'] = 'synth_LBO'
namelist['LBO']['t_start'] = 2.75
namelist['LBO']['t_rise'] = 0.3
namelist['LBO']['t_fall'] = 1.5
namelist['LBO']['n_particles'] = 1e18

# Change radial resolution from default:
namelist['dr_0']=1.0
namelist['dr_1']=0.1

# Change time resolution from default:
namelist['timing']['dt_increase'] = np.array([1.01, 1.])
namelist['timing']['dt_start'] = np.array([1e-5, 0.001])
namelist['timing']['steps_per_cycle'] = np.array([1,1])
namelist['timing']['times'] = np.array([2.749,2.8])

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# check radial grid:
#_ = aurora.create_radial_grid(namelist,plot=True)

# check time grid:
#_ = aurora.create_time_grid(namelist['timing'], plot=True)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -1e3 * asim.rhop_grid**5 # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

# plot radiation profiles over radius and time
aurora.slider_plot(asim.rvol_grid, asim.time_out, nz.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)


# weights for line integration
line_int = exc_pec_vals * asim.ne * interp1d(rhop_lineint, weights_lineint)(asim.rhop_grid)
spred_cs = line_int.T[:,None,:] * nz

aurora.slider_plot(asim.rvol_grid, asim.time_out, spred_cs.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)

# actual line integration
spred_synth = np.sum(spred_cs,axis=0)

fig,ax = plt.subplots()
ax.plot(asim.time_out, spred_synth.T)
ax.set_xlabel('time [s]')
ax.set_ylabel('signal [A.U.]')

aurora.slider_plot(asim.time_out, [1.], spred_synth[:,:,None],
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'SPRED signal [A.U.]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=False)
