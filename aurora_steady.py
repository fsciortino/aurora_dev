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
import copy

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

try: # pass any argument via the command line to show plots
    plot = len(sys.argv)>1
except:
    plot = False

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
examples_dir = '/home/sciortino/Aurora/examples'
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
rhop = kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
ne = kp['ne']['vals'] = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
Te = kp['Te']['vals'] = inputgacode['Te']*1e3  # keV --> eV

# set impurity species and sources rate to 0
imp = namelist['imp'] = 'Ar'
namelist['source_type'] = 'const'
namelist['source_rate'] = 0.0  # particles/s

# get charge state distributions from ionization equilibrium
atom_data = aurora.atomic.get_atom_data(imp,['scd','acd'])

# get fractional abundances on ne (cm^-3) and Te (eV) grid
logTe, fz = aurora.atomic.get_frac_abundances(
    atom_data, ne, Te, rho =rhop, plot=plot)

# eliminate annoying -ve numbers..
fz[fz<0] = 1e-99

# initial guess for steady state Ar charge state densities
_nz_init = ne[:,None] * fz

# Change radial resolution from default:
#namelist['dr_0']=0.2
#namelist['dr_1']=0.02

# Minimize boundary grid in simulation
namelist['bound_sep'] = 0.2
namelist['lim_sep'] = 0.1




################## Simulation time steps and duration settings ##################
n_rep = 100
dt = 1e-5

# Total time to run [s] will be approximated by nearest multiplier of n_rep*dt
max_sim_time = 50e-3
num_sims = int(max_sim_time/(n_rep*dt))
##################################################################################

# do only a few time steps per "run"
namelist['timing'] = {'dt_increase': np.array([1.0, 1.   ]),
                      'dt_start': np.array([dt, max_sim_time]),
                      'steps_per_cycle': np.array([1, 1]),
                      'times': np.array([0. , n_rep*dt])}

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

time_grid = copy.deepcopy(asim.time_grid)

# get initial guess for charge state profiles on the right radial grid
nz_init = interp1d(rhop, _nz_init, axis=0, bounds_error=False, fill_value=0.0)(asim.rhop_grid)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# run Aurora forward model and plot results
nz_new = asim.run_aurora(D_z, V_z, nz_init=nz_init, plot=False)[0]
nz_all = copy.deepcopy(nz_new)
nz_norm = nz_new[:,:,-1]/np.max(nz_new[:,:,-1]) 

for i in np.arange(num_sims):
    # Update time array
    namelist['timing']['times'] = np.array([(i+1)*n_rep*dt+dt, (i+2)*n_rep*dt])
    asim.setup_grids()

    # get charge state densities from latest time step
    nz_init_new = nz_all[:,:,-1]
    nz_new = asim.run_aurora(D_z, V_z, nz_init=nz_init_new, plot=False)[0]

    nz_all = np.dstack((nz_all, nz_new))
    time_grid = np.concatenate((time_grid, asim.time_grid))
    
    # check if normalized profiles have converged
    nz_norm_new = nz_new[:,:,-1]/np.max(nz_new[:,:,-1]) 
    if np.linalg.norm(nz_norm_new-nz_norm,ord=2)<1e-2:   # tolerance of 1%
        break
        
    nz_norm = copy.deepcopy(nz_norm_new)
    


# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rhop_grid, time_grid, nz_all.transpose(1,0,2),
                              xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz_all.shape[1])],
                              plot_sum=True)
