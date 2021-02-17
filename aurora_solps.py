'''Aurora-SOLPS coupling methods.

sciortino, 2021
'''
import pickle as pkl
import matplotlib.pyplot as plt
import MDSplus, os
import numpy as np
plt.ion()
from scipy.interpolate import interp1d
import aurora

from IPython import embed
plt.style.use('/home/sciortino/SPARC/sparc_plots.mplstyle')

def overplot_machine(shot, ax):
    # ----------------------------#
    # Overplot machine structures #
    # ----------------------------#
    MDSconn = MDSplus.Connection('alcdata.psfc.mit.edu')

    #pull cross-section from tree
    try:
        ccT = MDSconn.openTree('analysis',shot)
        path = '\\analysis::top.limiters.tiles:'
        xvctr = MDSconn.get(path+'XTILE').data()
        yvctr = MDSconn.get(path+'YTILE').data()
        nvctr = MDSconn.get(path+'NSEG').data()
        lvctr = MDSconn.get(path+'PTS_PER_SEG').data()
    except:
        raise ValueError('data load failed.')

    x = []
    y = []
    for i in range(nvctr):
        length = lvctr[i]
        xseg = xvctr[i,0:length]
        yseg = yvctr[i,0:length]
        x.extend(xseg)
        y.extend(yseg)
        if i != nvctr-1:
            x.append(None)
            y.append(None)

    x = np.array(x); y = np.array(y)

    for ii in np.arange(len(ax)):
        ax[ii].plot(x,y)


def compare_midplane_n0_with_expt(shot, rhop_solps, n0_solps_cm3):
    '''Compare midplane n0 and n0/ne between SOLPS-ITER and experimental Ly-alpha data.
    '''

    num_shade=10
    ff = 1./np.log(10.)

    # plot SOLPS midplane profiles
    fig, ax = plt.subplots(1,2, figsize=(12,6),sharex=True)

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

    ax[0].plot(rhop, np.log10(N1_prof), c='k')
    for ij in np.arange(num_shade):
        ax[0].fill_between(rhop, np.log10(N1_prof)+3*ff*N1_prof_unc/N1_prof*ij/num_shade,
                           np.log10(N1_prof)-3*ff*N1_prof_unc/N1_prof*ij/num_shade,
                           alpha=0.3*(1.-ij/num_shade), color='k')

    ax[1].plot(rhop, np.log10(N1_by_ne_prof),c='k', label='Ly-alpha (expt.)')
    for ij in np.arange(num_shade):
        ax[1].fill_between(rhop,np.log10(N1_by_ne_prof)+3*ff*N1_by_ne_prof_unc/N1_by_ne_prof*ij/num_shade,
                           np.log10(N1_by_ne_prof)-3*ff*N1_by_ne_prof_unc/N1_by_ne_prof*ij/num_shade,
                           alpha=0.3*(1.-ij/num_shade),
                           color='k')

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



from aurora.solps import solps_case

# I-mode:
#shots=[1080416025, 1101014029, 1101014030]

# L-mode:
#shots = [1100308004, 1101014006]

# H-mode:
#shots = [1100305023, 1101014019]
    
case='CMOD'
if case=='CMOD':
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
    so = solps_case(path, gfilepath, case_num=case_num, form=form)

elif case=='SPARC':
    path = '/home/sciortino/SPARC/V1E_LSN2_D+C'
    solps_run = 'P29MW_n1.65e20_Rout0.9_Y2pc'
    form='full'
    gfilepath = path+os.sep+'baserun'+os.sep+'V1E_geqdsk_LSN2'

    # load SOLPS results
    so = solps_case(path, gfilepath, solps_run=solps_run,form=form)
else:
    raise ValueError('Unrecognized case')


# fetch and process SOLPS output 
so.process_solps_data(plot=False)

max_mask_len=0.5 #07 if case=='CMOD' else None
use_triang=False

so.plot_2d_vals(np.log10(so.quants['nn']), use_triang=use_triang,
                label=so.labels['nn'],  #r'$log_{10}(n_n)$ [$10^{20}$ $m^{-3}$]'
                max_mask_len=max_mask_len)
if case=='CMOD':
    overplot_machine(shot, [plt.gca()]) # overplot machine tiles


# compare SOLPS results at midplane with FSA
rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS = so.get_radial_prof(quant='nn')

# ne on HFS and LFS must align!
rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so.get_radial_prof(quant='ne', plot=True)

# Obtain impurity charge state predictions from ioniz equilibrium
imp = 'C'
ne_cm3 = so.quants['ne'] *1e-6  #cm^-3
n0_cm3 = so.quants['nn'] *1e-6 #cm^-3
Te_eV = so.quants['Te']  # eV
filetypes=['acd','scd','ccd']
filenames = []
atom_data = aurora.get_atom_data(imp,filetypes,filenames)

logTe, fz, rate_coeffs = aurora.get_frac_abundances(
    atom_data,ne_cm3,Te_eV, n0_by_ne=n0_cm3/ne_cm3, plot=False)  # added n0

frac=0.01  # 1%
nz_cm3 = frac * ne_cm3[:,:,None] * fz  # (time,nZ,space) --> (R,nZ,Z)
nz_cm3 = nz_cm3.transpose(0,2,1)

# compute radiation density from each grid point
out = aurora.compute_rad(imp,nz_cm3, ne_cm3, Te_eV, n0=n0_cm3, Ti=Te_eV,
                         prad_flag=True,thermal_cx_rad_flag=True)

# plot total radiated power
so.plot_2d_vals(np.log10(out['tot']*1e3), use_triang=use_triang,
                label=r'$log_{10}(P_{rad})$ [$kW/m^3$]', max_mask_len=max_mask_len)
if case=='CMOD':
    overplot_machine(shot, [plt.gca()]) # overplot machine tiles

# plot total line radiated power
so.plot_2d_vals(np.log10(out['line_rad'].sum(1)*1e3), use_triang=use_triang,
                label=r'$log_{10}(P_{line,rad})$ [$kW/m^3$]',
                max_mask_len=max_mask_len)
if case=='CMOD':
    overplot_machine(shot, [plt.gca()]) # overplot machine tiles

    compare_midplane_n0_with_expt(shot,  rhop_LFS, neut_LFS*1e-6)



# compare neutral D2 pressure among CMOD shots

tmin=0.8
tmax=1.0
from lyman_data import *

for shot in [1100308004,1100305023,1080416025]:
    out = get_CMOD_gas_fueling(shot, tmin=tmin, tmax=tmax, get_rate=False, plot=True)
    plt.gca().set_title(f'shot {shot}')
    
    P_D2 = get_CMOD_var('p_D2',shot, tmin=tmin, tmax=tmax, plot=True)
    plt.gca().set_title(f'shot {shot}')
    print(f'shot {shot} had P_D2={P_D2}')
