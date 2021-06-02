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

device='ITER' #'CMOD' #'ITER' #'CMOD' #'ITER'
imp = 'Fe' #'W' #'Fe' #'Ar' #'W' #'Ar' #'W' #'Ar'
Dval = 10e4 #10e4 if device=='ITER' else 5e4

if device=='ITER':
    case='/home/sciortino/ITER/ITER.baseline'
    inputgacode = omfit_gapy.OMFITinputprofiles(case+'/input.profiles')
    geqdsk = omfit_eqdsk.OMFITgeqdsk('/home/sciortino/ITER/gfile_iter')
else:  # CMOD
    # Use gfile and statefile in local directory:
    examples_dir = '/home/sciortino/Aurora/examples'
    geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
    inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')


# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
rhop = kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
ne_cm3 = kp['ne']['vals'] = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
Te_eV = kp['Te']['vals'] = inputgacode['Te']*1e3  # keV --> eV

# set impurity species and sources rate
namelist['imp'] = imp
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

namelist['timing']['times'][1] = 1.0

if imp=='Ar':
    superstages = [0,1,14,15,16,17,18]
if imp=='Fe':
    #superstages = [0,1,2,8,16,17,18,19,20,21,22,23,24,25,26]
    superstages = [0,1,16,17,18,19,20,21,22,23,24,25,26]
elif imp=='W':
    #superstages = np.array([2*n**2 for n in np.arange(7)]) #np.concatenate((np.array([0.,]), np.arange(10,50,3)))
    #superstages = np.array([0,1,2,8,15,18,32,50,74])
    superstages = np.concatenate((np.array([0,1]), np.arange(50,71,3))) #, np.array([74,])))
    superstages = np.concatenate((np.array([0]), np.arange(50,71,3))) 
    
########
# first run WITHOUT superstages
namelist['superstages'] = []

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
#D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
#V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

D_z = Dval * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = 0.0 * asim.rhop_grid**6 # * np.ones(len(asim.rvol_grid)) # cm/s

#-100e2

#D_z = np.tile(D_z, (len(namelist['superstages'])+1,1,1)).T
#V_z = np.tile(V_z, (len(namelist['superstages'])+1,1,1)).T

#D_z = np.tile(D_z, (asim.Z_imp+1,1,1)).T
#V_z = np.tile(V_z, (asim.Z_imp+1,1,1)).T

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, times_DV=[1.0,], unstage=True, plot=False)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out


# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rhop_grid, asim.time_grid, nz.transpose(1,0,2),
                              xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True)#, x_line=asim.rvol_lcfs)


########
# now choose superstages: always include 0 and 1!
namelist['superstages'] = superstages

# set up aurora again, this time with superstages
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, times_DV=[1.0,], unstage=True, plot=False)

# extract densities and particle numbers in each simulation reservoir
nzs, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_grid, nzs.transpose(1,0,2),
                              xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nzs.shape[1])],
                              plot_sum=True)#, x_line=asim.rvol_lcfs)


# compare at last slice
ls_cycle= aurora.get_ls_cycle()

fig = plt.figure()
fig.set_size_inches(9,6, forward=True)
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 2, fig=fig)
a_legend.axis('off')

if imp=='Fe' and device=='ITER':
    left, bottom, width, height = [0.4, 0.67, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    
for cs in np.arange(nz.shape[1]):
    if cs<15:
        continue
    ls = next(ls_cycle) 
    a_plot.plot(asim.rhop_grid, nz[:,cs,-1],ls, lw=1.0)
    a_plot.plot(asim.rhop_grid, nzs[:,cs,-1],ls, lw=2.)
    a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')

    if imp=='Fe' and device=='ITER':
        ax2.plot(asim.rhop_grid, nz[:,cs,-1],ls, lw=1.0)
        ax2.plot(asim.rhop_grid, nzs[:,cs,-1],ls, lw=2.)
        
a_plot.set_xlabel(r'$\rho_p$')
a_plot.set_ylabel(r'$n_z$ [A.U.]')

if imp=='Fe' and device=='ITER':
    ax2.set_xlabel(r'$\rho_p$')
    #ax2.set_ylabel(r'$n_z$ [A.U.]')
    ax2.set_xlim([0.88,0.95])
    ax2.set_xticks([0.88, 0.91, 0.94])
    
    # good scales for AURORA paper, Fe plot
    #a_plot.set_ylim([None, 3.8e8])
    #ax2.set_ylim([2.85e8,2.92e8])
    # ax2.set_yticks([2.86e8,2.91e8])
    
a_legend.legend(loc='best', ncol=1).set_draggable(True)




# compare at last slice
ls_cycle= aurora.get_ls_cycle()

fig = plt.figure()
fig.set_size_inches(9,6, forward=True)
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 2, fig=fig)
a_legend.axis('off')

left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

css=0
for cs in np.arange(nz.shape[1]):
    ls = next(ls_cycle) 
    a_plot.plot(asim.rhop_grid, nz[:,cs,-1],ls, lw=1.0)
    ax2.plot(asim.rhop_grid, nz[:,cs,-1],ls, lw=1.0)
    
    if cs in superstages:
        a_plot.plot(asim.rhop_grid, nzs[:,css,-1],ls, lw=2.)
        a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
        ax2.plot(asim.rhop_grid, nzs[:,css,-1],ls, lw=2.)
        css+=1
    
a_plot.set_xlabel(r'$\rho_p$')
a_plot.set_ylabel(r'$n_z$ [A.U.]')
ax2.set_xlim([0.8,1.0])

a_legend.legend(loc='best', ncol=1).set_draggable(True)





if imp=='W':
    #### only show some key charge states in the comparison

    cs_to_compare = np.arange(50,71,3)
    #cs_to_compare = np.arange(50,70,2) # np.arange(40,70,3)
    ls_cycle= aurora.get_ls_cycle()

    fig = plt.figure()
    fig.set_size_inches(9,6, forward=True)
    a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
    a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 2, fig=fig)
    a_legend.axis('off')

    css=0
    for cs in np.arange(nz.shape[1]):
        if cs in cs_to_compare:
            ls = next(ls_cycle)
            a_plot.plot(asim.rhop_grid, nz[:,cs,-1],ls, lw=1.0)
            a_plot.plot(asim.rhop_grid, nzs[:,css,-1],ls, lw=2.)
            a_legend.plot([],[], ls, label=imp+f'$^{{{cs}+}}$')
        css+=1

    a_plot.set_xlabel(r'$\rho_p$')
    a_plot.set_ylabel(r'$n_z$ [A.U.]')

    a_legend.legend(loc='best', ncol=1).set_draggable(True)


    #### Check radiation inaccuracy

    # rescale nz profile to have W_frac fractional abundance on axis
    W_frac = 1.5e-5
    scale = nz[0,:,-1].sum()/asim.ne[0,0]/W_frac
    
    res_f = aurora.radiation_model(imp, asim.rhop_grid, asim.ne[0], asim.Te[0], geqdsk,
                                      nz_cm3=scale*nz[...,-1], plot=False)
    res_s = aurora.radiation_model(imp, asim.rhop_grid, asim.ne[0], asim.Te[0], geqdsk,
                                       nz_cm3=scale*nzs[...,-1], plot=False)

    fig, ax = plt.subplots()
    ax.plot(res_f['rhop'], res_f['rad_tot']/1e6, c='r', label='full model')
    ax.plot(res_s['rhop'], res_s['rad_tot']/1e6, c='b', label='reduced model (superstaging)')
    ax.set_xlabel(r'$\rho_p$')
    ax.set_ylabel(r'$P_{rad}$ [MW]')
    ax.legend(loc='best').set_draggable(True)
    plt.tight_layout()

    '''
    # synchrotron is negligible
    psin_ref = geqdsk['fluxSurfaces']['geo']['psin']
    rhop_ref = np.sqrt(psin_ref) # sqrt(norm. pol. flux)
    
    ne_interp = interp1d(asim.rhop_grid, asim.ne[0])(rhop_ref)
    Te_interp = interp1d(asim.rhop_grid, asim.Te[0])(rhop_ref)
    
    aurora.sync_rad(5.7, #np.abs(geqdsk['fluxSurfaces']['midplane']['Bt'])
                    ne_interp, Te_interp,
                    geqdsk['fluxSurfaces']['geo']['a']*100,
                    geqdsk['fluxSurfaces']['R0']*100)
    '''
