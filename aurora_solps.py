'''Aurora-SOLPS coupling methods.

sciortino, 2021
'''
import pickle as pkl
import matplotlib.pyplot as plt
import MDSplus, os, copy, sys
import numpy as np
plt.ion()
from scipy.interpolate import interp1d
import aurora
from heapq import nsmallest
from IPython import embed
#plt.style.use('/home/sciortino/SPARC/sparc_plots.mplstyle')
from scipy import constants
from scipy.interpolate import griddata, RectBivariateSpline, interp1d
import profiletools

import matplotlib.colors as colorsMPL
import math
import os.path

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

if '/home/sciortino/tools3/' not in sys.path:
    sys.path.append('/home/sciortino/tools3/')

from plot_cmod_machine import overplot_machine    

import aurora


import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

def plot_lya(shot, ax, c='k', label=''):

    num_shade=10
    ff = 1./np.log(10.)

    # plot Ly-alpha data
    with open(f'/home/sciortino/tools3/neutrals/lyman_data_{shot}.pkl','rb') as f:
        out_Lya = pkl.load(f)

    rhop,roa,R, N1_prof,N1_prof_unc,ne_prof,ne_prof_unc,Te_prof,Te_prof_unc = out_Lya

    # mask out nan's
    mask = ~np.isnan(N1_prof)
    rhop=rhop[mask]; roa=roa[mask]; R=R[mask]
    N1_prof=N1_prof[mask]; N1_prof_unc=N1_prof_unc[mask];
    ne_prof=ne_prof[mask]; ne_prof_unc=ne_prof_unc[mask]
    Te_prof=Te_prof[mask]; Te_prof_unc=Te_prof_unc[mask]
    
    N1_by_ne_prof = N1_prof/ne_prof
    
    # ne uncertainty in the SOL also goes to ne<0.... ignore it
    N1_by_ne_prof_unc = np.sqrt((N1_prof_unc/ne_prof)**2) #+(N1_prof/ne_prof**2)**2*ne_prof_unc**2)  

    ax[0].plot(rhop, np.log10(N1_prof), c=c)
    for ij in np.arange(num_shade):
        ax[0].fill_between(rhop, np.log10(N1_prof)+3*ff*N1_prof_unc/N1_prof*ij/num_shade,
                           np.log10(N1_prof)-3*ff*N1_prof_unc/N1_prof*ij/num_shade,
                           alpha=0.3*(1.-ij/num_shade), color=c)

    ax[1].plot(rhop, np.log10(N1_by_ne_prof),c=c, label='Ly-alpha (expt.) '+label)
    for ij in np.arange(num_shade):
        ax[1].fill_between(rhop,np.log10(N1_by_ne_prof)+3*ff*N1_by_ne_prof_unc/N1_by_ne_prof*ij/num_shade,
                           np.log10(N1_by_ne_prof)-3*ff*N1_by_ne_prof_unc/N1_by_ne_prof*ij/num_shade,
                           alpha=0.3*(1.-ij/num_shade),
                           color=c)


def compare_neutral_profs(expt=True):

    fig, ax = plt.subplots(1,2, figsize=(12,6),sharex=True)

    if expt:
        # load and plot Ly-a data
        plot_lya(1100308004, ax, c='k', label='L-mode')
        plot_lya(1080416025, ax, c='b', label='I-mode')
        plot_lya(1100305023, ax, c='r', label='EDA H-mode')
    else:
        # compare SOLPS cases

        # L-mode
        shot = 1100308004; solps_run='Attempt14'  # L-mode (new)
        path = f'/home/sciortino/SOLPS/full_CMOD_runs/Lmode_1100308004/'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1100308004.01049'
        so_L = aurora.solps_case(path, gfilepath, solps_run=solps_run,form='full')

        rhop_fsa, nn_fsa, rhop_LFS, nn_LFS, rhop_HFS, nn_HFS = so_L.get_radial_prof(so_L.quants['nn'], plot=False)
        rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so_L.get_radial_prof(so_L.quants['ne'], plot=False) 
        ax[0].plot(rhop_LFS, np.log10(nn_LFS*1e-6), lw=3, c='k')
        ax[1].plot(rhop_LFS, np.log10(nn_LFS/ne_LFS), lw=3, c='k', label='SOLPS-ITER L-mode')

        # I-mode
        shot = 1080416025; solps_run='Attempt15N'
        path = '/home/sciortino/SOLPS/full_CMOD_runs/Imode_1080416025'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1080416025.01000'
        so_I = aurora.solps_case(path, gfilepath, solps_run=solps_run,form='full')

        rhop_fsa, nn_fsa, rhop_LFS, nn_LFS, rhop_HFS, nn_HFS = so_I.get_radial_prof(so_I.quants['nn'], plot=False)
        rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so_I.get_radial_prof(so_I.quants['ne'], plot=False) 
        ax[0].plot(rhop_LFS, np.log10(nn_LFS*1e-6), lw=3, c='b')
        ax[1].plot(rhop_LFS, np.log10(nn_LFS/ne_LFS), lw=3, c='b', label='SOLPS-ITER I-mode')

        # H-mode
        shot = 1100305023; solps_run='Attempt23' # H-mode
        path = '/home/sciortino/SOLPS/full_CMOD_runs/Hmode_1100305023'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1100305023.01075'
        so_H = aurora.solps_case(path, gfilepath, solps_run=solps_run,form='full')

        rhop_fsa, nn_fsa, rhop_LFS, nn_LFS, rhop_HFS, nn_HFS = so_H.get_radial_prof(so_H.quants['nn'], plot=False)
        rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so_H.get_radial_prof(so_H.quants['ne'], plot=False) 
        ax[0].plot(rhop_LFS, np.log10(nn_LFS*1e-6), lw=3, c='r')
        ax[1].plot(rhop_LFS, np.log10(nn_LFS/ne_LFS), lw=3, c='r', label='SOLPS-ITER H-mode')
        
        
    ax[0].set_ylabel(r'$log_{10}(n_0$ [$cm^{-3}$])')
    ax[1].set_ylabel(r'$log_{10}(n_0/n_e)$')
    ax[1].set_xlabel(r'$\rho_p$')
    ax[0].set_xlabel(r'$\rho_p$')

    # set convenient limts
    #ax[0].set_xlim([np.min(rhop_LFS),np.max(rhop_LFS)])
    ax[0].set_xlim([0.955, 1.0])
    ax[0].set_ylim([8.5,11])
    ax[1].set_ylim([-5.5,-2.5])
    ax[1].legend(loc='best').set_draggable(True)
    fig.tight_layout()


def compare_midplane_n0_with_expt(shot, rhop_solps, n0_solps_cm3):
    '''Compare midplane n0 and n0/ne between SOLPS-ITER and experimental Ly-alpha data.
    '''

    fig, ax = plt.subplots(1,2, figsize=(12,6),sharex=True)

    # load and plot Ly-a data
    plot_lya(shot, ax)

    # Now add SOLPS result
    n0_solps_interp = interp1d(rhop_solps, n0_solps_cm3, bounds_error=False)(rhop)
    ax[0].plot(rhop, np.log10(n0_solps_interp), lw=3, c='r')
    ax[1].plot(rhop, np.log10(n0_solps_interp/ne_prof), lw=3, c='r', label='SOLPS-ITER')
    
    ax[0].set_ylabel(r'$log_{10}(n_0$ [$cm^{-3}$])')
    ax[1].set_ylabel(r'$log_{10}(n_0/n_e)$')
    ax[1].set_xlabel(r'$\rho_p$')
    ax[0].set_xlabel(r'$\rho_p$')

    # set convenient limts
    ax[0].set_xlim([np.min(rhop),np.max(rhop)])
    ax[0].set_ylim([8,13])
    ax[1].set_ylim([-6,-1])
    ax[1].legend(loc='best').set_draggable(True)
    fig.tight_layout()




def compare_to_TS(shot):
    '''Function to compare SOLPS results to TS data from CMOD.
    '''
    if shot== 1120917011:
        # L-mode (J.Rice)
        solps_run='Attempt75' 
        path = f'/home/sciortino/SOLPS/full_CMOD_runs/Lmode_1120917011/'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g{shot}.00999_981'  # hard-coded

    elif shot==1100308004:
        # L-mode (new)
        solps_run='Attempt14'
        path = f'/home/sciortino/SOLPS/full_CMOD_runs/Lmode_1100308004/'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1100308004.01049'

    elif shot==1100305023:
        # H-mode
        solps_run='Attempt23' # H-mode
        path = '/home/sciortino/SOLPS/full_CMOD_runs/Hmode_1100305023'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1100305023.01075'

    elif shot==1080416025:
        # I-mode
        solps_run='Attempt15N'
        path = '/home/sciortino/SOLPS/full_CMOD_runs/Imode_1080416025'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1080416025.01000'

    # load SOLPS results
    so = aurora.solps_case(path, gfilepath, solps_run=solps_run,form='full')
    
    # assumed time-average range:
    t0 = 0.9
    t1 = 1.1

    # Load experimental Thomson data   
    p_ne = profiletools.neETS(shot, t_min=t0, t_max=t1, abscissa='RZ')
    p_Te = profiletools.TeETS(shot, t_min=t0, t_max=t1, abscissa='RZ')
    
    # Combine TS data from multiple shots that are supposedly identical
    #Imode_shots = [1080416025, 1101014029, 1101014030]
    #Lmode_shots = [1100308004, 1101014006]
    #Hmode_shots = [1100305023, 1101014019]
    
    if shot==1080416025:
        p_ne2 = profiletools.neETS(1101014029, t_min=t0, t_max=t1, abscissa='RZ')
        p_ne.add_profile(p_ne2)
        #p_ne3 = profiletools.neETS(1101014030, t_min=t0, t_max=t1, abscissa='RZ') # no ETS data
        #p_ne.add_profile(p_ne3)

        p_Te2 = profiletools.TeETS(1101014029, t_min=t0, t_max=t1, abscissa='RZ')
        p_Te.add_profile(p_Te2)
        #p_Te3 = profiletools.TeETS(1101014030, t_min=t0, t_max=t1, abscissa='RZ')  # no ETS data
        #p_Te.add_profile(p_Te3)
        
    elif shot==1100308004:
        p_ne2 = profiletools.neETS(1101014006, t_min=t0, t_max=t1, abscissa='RZ')
        p_ne.add_profile(p_ne2)

        p_Te2 = profiletools.TeETS(1101014006, t_min=t0, t_max=t1, abscissa='RZ')
        p_Te.add_profile(p_Te2)

    elif shot==1100305023:
        p_ne2 = profiletools.neETS(1101014019, t_min=t0, t_max=t1, abscissa='RZ')
        p_ne.add_profile(p_ne2)

        p_Te2 = profiletools.TeETS(1101014019, t_min=t0, t_max=t1, abscissa='RZ')
        p_Te.add_profile(p_Te2)
    
    else:
        pass
    
    p_ne.time_average(weighted=True)
    p_Te.time_average(weighted=True)
    
    min_ne_err=0.01
    min_Te_err=0.01
    max_ne_err = 0.2
    max_Te_err = 0.3
    import fit_2D
    fit_2D.clean_profs(shot,p_ne, p_Te, min_ne_err, max_ne_err, min_Te_err, max_Te_err)
    
    fig,ax = plt.subplots(figsize=(8,10))
    overplot_machine(shot, ax)
    so.geqdsk.plot(only2D=True, ax=ax, color='r')
    ax.plot(p_ne.X[:,0], p_ne.X[:,1], '*')
    ax.set_xlabel(r'$R$ $[m]$')
    ax.set_ylabel(r'$Z$ $[m]$')
    


    # get vertical slice at Thomson points    
    ne_pts_solps = griddata((so.R.flatten(),so.Z.flatten()), so.quants['ne'].flatten(),
                       (p_ne.X[:,0], p_ne.X[:,1]), method='linear')
    Te_pts_solps = griddata((so.R.flatten(),so.Z.flatten()), so.quants['Te'].flatten(),
                       (p_Te.X[:,0], p_Te.X[:,1]), method='linear')

    # Now get midplane quantities from SOLPS
    rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so.get_radial_prof(so.quants['ne'], plot=False)
    rhop_fsa, Te_fsa, rhop_LFS, Te_LFS, rhop_HFS, Te_HFS = so.get_radial_prof(so.quants['Te'], plot=False)

    def get_rhop_RZ(R,Z, geqdsk):
        '''Find rhop at every R,Z [m] based on the equilibrium in the geqdsk dictionary.
        '''
        return RectBivariateSpline(geqdsk['AuxQuantities']['Z'],
                                   geqdsk['AuxQuantities']['R'],
                                   geqdsk['AuxQuantities']['RHOpRZ']).ev(Z,R)

    # obtain rhop of point where TS measurements are made
    rhop_ne_meas = get_rhop_RZ(p_ne.X[:,0], p_ne.X[:,1], so.geqdsk)
    rhop_Te_meas = get_rhop_RZ(p_Te.X[:,0], p_Te.X[:,1], so.geqdsk)

    # Now find R value along TS measurement chord that would correspond to LFS midplane rhop's
    ne_wrong_pts = interp1d(rhop_LFS, ne_LFS, bounds_error=False)(rhop_ne_meas)
    Te_wrong_pts = interp1d(rhop_LFS, Te_LFS, bounds_error=False)(rhop_Te_meas)
    

    # plot comparison between SOLPS and TS
    fig,ax = plt.subplots(2,1,figsize=(10,6))
    ax[0].plot(p_ne.X[:,1], ne_pts_solps, '.', label='SOLPS-ITER (correct)')
    ax[0].errorbar(p_ne.X[:,1], p_ne.y*1e20, p_ne.err_y*1e20, fmt='.', label='ETS (expt.)')
    ax[0].plot(p_ne.X[:,1], ne_wrong_pts, '.', label='SOLPS-ITER mapped from midplane')
    ax[0].set_xlabel('Z [m]')
    ax[0].set_ylabel(r'$n_e$ $[m^{-3}]$')
    ax[1].plot(p_Te.X[:,1], Te_pts_solps, '.', label='SOLPS-ITER (correct)')
    ax[1].errorbar(p_Te.X[:,1], p_Te.y*1e3, p_Te.err_y*1e3, fmt='.', label='ETS (expt.)')
    ax[1].plot(p_Te.X[:,1], Te_wrong_pts, '.', label='SOLPS-ITER')
    ax[1].set_xlabel('Z [m]')
    ax[1].set_ylabel(r'$T_e$ $[eV]$')
    ax[0].legend(loc='best').set_draggable(True)
    plt.tight_layout()

    
    

    



if __name__=='__main__':
                  
    device='CMOD_full' #'ITER' #'CMOD_full' #'SPARC'
    if device=='CMOD':
        # L-mode (old)
        #shot = 1120917011; case_num = 68 #4 
        #path = f'/home/sciortino/SOLPS/RR_{shot}_attempts/Attempt{case_num}/Output/'
        #gfilepath = f'/home/sciortino/EFIT/gfiles/g{shot}.00999_981'  # hard-coded

        # L-mode (new)
        #shot = 1100308004; case_num = 14 
        #path = f'/home/sciortino/SOLPS/RR_Lmode_attempt{case_num}/Output/'
        #gfilepath = f'/home/sciortino/EFIT/gfiles/g1100308004.01049'

        # H-mode
        #shot = 1100305023; case_num = 23 
        #path = f'/home/sciortino/SOLPS/RR_Hmode_attempt{case_num}/Output/'
        #gfilepath = f'/home/sciortino/EFIT/gfiles/g1100305023.01075'

        # I-mode
        #shot = 1080416025; case_num = 15  # H-mode
        #path = f'/home/sciortino/SOLPS/RR_Imode_attempt{case_num}_old/Output/'
        #gfilepath = f'/home/sciortino/EFIT/gfiles/g1080416025.01000'

        # I-mode (new)
        shot = 1080416025; case_num = '15' # '15N'
        path = f'/home/sciortino/SOLPS/RR_Imode_attempt{case_num}/Output/'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1080416025.01000'

        # load lyman-alpha data
        with open(f'/home/sciortino/tools3/neutrals/lyman_data_{shot}.pkl','rb') as f:
            out_expt = pkl.load(f)
        rhop,roa,R, N1_prof,N1_prof_unc,ne_prof,ne_prof_unc,Te_prof,Te_prof_unc = out_expt

        form='extracted'

        # load SOLPS results
        so = aurora.solps_case(path, gfilepath, case_num=case_num, form=form)

    elif device=='CMOD_full':

        # L-mode (J.Rice)
        #shot = 1120917011; solps_run='Attempt75' 
        #path = f'/home/sciortino/SOLPS/full_CMOD_runs/Lmode_1120917011/'
        #gfilepath = f'/home/sciortino/EFIT/gfiles/g{shot}.00999_981'  # hard-coded

        # L-mode (new)
        shot = 1100308004; solps_run='Attempt14'
        path = f'/home/sciortino/SOLPS/full_CMOD_runs/Lmode_1100308004/'
        gfilepath = f'/home/sciortino/EFIT/gfiles/g1100308004.01049'

        # H-mode
        #shot = 1100305023; solps_run='Attempt23' # H-mode
        #path = '/home/sciortino/SOLPS/full_CMOD_runs/Hmode_1100305023'
        #gfilepath = f'/home/sciortino/EFIT/gfiles/g1100305023.01075'

        # I-mode
        #shot = 1080416025; solps_run='Attempt15N'
        #path = '/home/sciortino/SOLPS/full_CMOD_runs/Imode_1080416025'
        #path = f'/home/sciortino/SOLPS/RR_Imode_attempt{case_num}_old/Output/'
        #gfilepath = f'/home/sciortino/EFIT/gfiles/g1080416025.01000'

        # load lyman-alpha data
        with open(f'/home/sciortino/tools3/neutrals/lyman_data_{shot}.pkl','rb') as f:
            out_expt = pkl.load(f)
        rhop,roa,R, N1_prof,N1_prof_unc,ne_prof,ne_prof_unc,Te_prof,Te_prof_unc = out_expt

        # load SOLPS results
        so = aurora.solps_case(path, gfilepath, solps_run=solps_run,form='full')

    elif device=='SPARC':
        path = '/home/sciortino/SPARC/V1E_LSN2_D+C'
        solps_run = 'P29MW_n1.65e20_Rout0.9_Y2pc_NPup'
        form='full'
        gfilepath = path+os.sep+'baserun'+os.sep+'V1E_geqdsk_LSN2'

        # load SOLPS results
        so = aurora.solps_case(path, gfilepath, solps_run=solps_run,form=form)

    elif device=='ITER':
        path = '/home/sciortino/ITER/iter_solps_jl'
        solps_run = 'orig_D1.95e23_Ne2.00e20.done.ids'
        form='full'
        gfilepath = '/home/sciortino/ITER/gfile_iter'

        so = aurora.solps_case(path, gfilepath, solps_run=solps_run,form=form)
    else:
        raise ValueError('Unrecognized case')



    # plot some important fields
    fig,axs = plt.subplots(1,4, figsize=(20,6),sharex=True) 
    ax = axs.flatten()
    so.plot2d_b2(so.quants['ne'], ax=ax[0], scale='log', label=so.labels['ne'])#; ax[0].axis('scaled')
    so.plot2d_b2(so.quants['Te'], ax=ax[1], scale='linear', label=so.labels['Te'])#; ax[1].axis('scaled')
    so.plot2d_b2(so.quants['nn'], ax=ax[2], scale='log', label=so.labels['nn'])#; ax[2].axis('scaled')
    so.plot2d_b2(so.quants['Tn'], ax=ax[3], scale='linear', label=so.labels['Tn'])#; ax[3].axis('scaled')
    plt.tight_layout()

    fig,axs = plt.subplots(1,2, figsize=(15,8),sharex=True) 
    ax = axs.flatten()
    so.plot2d_b2(so.b2fstate['na'][0,:,:], ax=ax[0], scale='log', label=so.labels['nn'])#; ax[0].axis('scaled')
    nn2 = so.fort44['dab2'][:,:,0].T
    #nn2[nn2==0.0] = nsmallest(2,np.unique(nn2.flatten()))[1]
    so.plot2d_b2(nn2, ax=ax[1], scale='log', label=so.labels['nn'])#; ax[2].axis('scaled')
    plt.tight_layout()


    #so.plot2d_b2(so.quants['nn'], label=so.labels['nn'],
    #             lb=np.quantile(so.quants['nn'], 0.1),
    #             ub=np.quantile(so.quants['nn'], 0.9))


    # comparison of neutrals in fort.44 and fort.46
    so.plot2d_eirene(so.fort46['pdena'][:,0]*1e6, scale='log', label=so.labels['nn'])

    so.plot2d_b2(so.fort44['dab2'][:,:,0].T, label=so.labels['nn'])


    if device=='CMOD':
        overplot_machine(shot, [plt.gca()]) # overplot machine tiles


    # compare SOLPS results at midplane with FSA
    rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS = so.get_radial_prof(so.quants['nn'], plot=True)

    # ne on HFS and LFS should align, but grid definitions cause some misalignment
    rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so.get_radial_prof(so.quants['ne'], plot=True)

    fig,ax = plt.subplots()
    ax.semilogy(rhop_LFS, neut_LFS/ne_LFS)
    ax.set_xlabel(r'$\rho_p$')
    ax.set_ylabel(r'$n_0/n_e$')

    # Obtain impurity charge state predictions from ioniz equilibrium
    imp = 'C'
    ne_cm3 = so.quants['ne'] *1e-6  #cm^-3
    n0_cm3 = so.quants['nn'] *1e-6 #cm^-3
    Te_eV = copy.deepcopy(so.quants['Te']) # eV
    Te_eV[Te_eV<1.]= 1.0
    filetypes=['acd','scd','ccd']
    atom_data = aurora.get_atom_data(imp,filetypes)

    logTe, fz, rate_coeffs = aurora.get_frac_abundances(
        atom_data,ne_cm3,Te_eV, n0_by_ne=n0_cm3/ne_cm3, plot=False)  # added n0

    frac=0.01  # 1%
    nz_cm3 = frac * ne_cm3[:,:,None] * fz  # (time,nZ,space) --> (R,nZ,Z)
    nz_cm3 = nz_cm3.transpose(0,2,1)

    # compute radiation density from each grid point
    out = aurora.compute_rad(imp,nz_cm3, ne_cm3, Te_eV, n0=n0_cm3, Ti=Te_eV,
                             prad_flag=True,thermal_cx_rad_flag=True)


    # plot total radiated power
    so.plot2d_b2(out['tot']*1e3, scale='log', label=r'$P_{rad}$ [$kW/m^3$]')

    if device=='CMOD':
        overplot_machine(shot, [plt.gca()]) # overplot machine tiles

    # plot total line radiated power
    so.plot2d_b2(out['line_rad'].sum(1)*1e3, scale='log', label=r'$P_{line,rad}$ [$kW/m^3$]')

    if device=='CMOD':
        overplot_machine(shot, [plt.gca()]) # overplot machine tiles

        compare_midplane_n0_with_expt(shot,  rhop_LFS, neut_LFS*1e-6)



    # compare neutral D2 pressure among CMOD shots

    # tmin=0.8
    # tmax=1.0
    # from lyman_data import *

    # for shot in [1100308004,1100305023,1080416025]:
    #     out = get_CMOD_gas_fueling(shot, tmin=tmin, tmax=tmax, get_rate=False, plot=True)
    #     plt.gca().set_title(f'shot {shot}')

    #     P_D2 = get_CMOD_var('p_D2',shot, tmin=tmin, tmax=tmax, plot=True)
    #     plt.gca().set_title(f'shot {shot}')
    #     print(f'shot {shot} had P_D2={P_D2}')






    #plt.close('all')
    '''
    ###### Comparison with KN1D #####
    rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so.get_radial_prof(so.quants['ne'], plot=False)
    rhop_fsa, Te_fsa, rhop_LFS, Te_LFS, rhop_HFS, Te_HFS = so.get_radial_prof(so.quants['Te'], plot=False)
    rhop_fsa, nm_fsa, rhop_LFS, nm_LFS, rhop_HFS, nm_HFS = so.get_radial_prof(so.quants['nm'], plot=False)
    rhop_fsa, Tm_fsa, rhop_LFS, Tm_LFS, rhop_HFS, Tm_HFS = so.get_radial_prof(so.quants['Tm'], plot=False)

    # pressure near the wall
    #p_H2_Pa = nm_LFS[-1]*Tm_LFS[-1]*constants.e # Pascals
    p_H2_Pa = np.nanmax(nm_LFS)*3.0*constants.e # Pascals
    rhop_edge = rhop_LFS[-1]

    # estimate connection lengths from the EFIT g-EQDSK
    from omfit_classes import omfit_eqdsk
    geqdsk = omfit_eqdsk.OMFITgeqdsk(gfilepath)
    clen_divertor_m, clen_limiter_m = aurora.estimate_clen(geqdsk)
    clen_divertor_cm = clen_divertor_m*1e2 
    clen_limiter_cm = clen_limiter_m*1e2

    # Estimate radial separation of boundary to separatrix and limiter to separatrix.
    #bound_sep_cm, lim_sep_cm = aurora.grids_utils.estimate_boundary_distance(shot, 'CMOD', 1000.)
    Rsep = aurora.rad_coord_transform(1.0,'rhop','Rmid', geqdsk)
    Rwall = aurora.rad_coord_transform(np.max(rhop_LFS),'rhop','Rmid', geqdsk)
    bound_sep_cm = (Rwall - Rsep)*1e2
    lim_sep_cm = bound_sep_cm*2./3. # doesn't matter, only for plotting

    # convert pressure from Pa to mTorr
    p_H2_mTorr= p_H2_Pa/133.32*1e3  # https://en.wikipedia.org/wiki/Torr

    if device.startswith('CMOD'):
        if '/home/sciortino/tools3/neutrals' not in sys.path:
            sys.path.append('/home/sciortino/tools3/neutrals')

        import lyman_data
        p_H2_mTorr_expt = lyman_data.get_CMOD_var(var='p_D2',shot=shot, tmin=0.9,tmax=1.1, plot=False)
        print('Comparison of EIRENE vs. expt H2 pressure:')
        print(f'EIRENE: {p_H2_mTorr:.3f} [mTorr]')
        print(f'expt: {p_H2_mTorr_expt:.3f} [mTorr]')

    # compare p_H2_mTorr with midplane pressure measurement on CMOD
    innermost_rmid_cm=5.0

    mask=~np.isnan(ne_LFS)
    kn1d_res = aurora.run_kn1d(rhop_LFS[mask],
                               ne_LFS[mask]*1e-6,  # aurora.kn1d input in cm^{-3}
                               Te_LFS[mask], Te_LFS[mask],  # aurora.kn1d inputs in eV
                               geqdsk, p_H2_mTorr, clen_divertor_cm, clen_limiter_cm,
                               bound_sep_cm, lim_sep_cm, innermost_rmid_cm, plot_kin_profs=True)

    # series of plots to visualize (processed) KN1D output
    aurora.kn1d.plot_overview(kn1d_res)     # overview of inputs and outputs
    aurora.kn1d.plot_exc_states(kn1d_res)   # excited states
    aurora.kn1d.plot_emiss(kn1d_res)        # Ly-a and D-a emission profiles

    # comparison of SOLPS neutral LFS profile and KN1D one
    rhop_fsa, nn_fsa, rhop_LFS, nn_LFS, rhop_HFS, nn_HFS = so.get_radial_prof(so.quants['nn'], plot=False)

    rhop_kn1d = aurora.rad_coord_transform(kn1d_res['out']['rwall_m'] - kn1d_res['out']['xh'][::-1], 'rmid', 'rhop', geqdsk)

    fig,ax = plt.subplots()
    ax.semilogy(rhop_kn1d, kn1d_res['out']['nh'], label='KN1D')
    ax.semilogy(rhop_LFS, nn_LFS, label='EIRENE')
    ax.set_xlabel(r'$\rho_p$')
    ax.set_ylabel(r'$n_n$ [m$^{-3}$]')
    ax.legend(loc='best').set_draggable(True)
    '''




