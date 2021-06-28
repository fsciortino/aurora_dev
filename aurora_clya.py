import aurora
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
#from numpy import *
import numpy as np
plt.ion()
import re

def overplot_spectra(pec_info, Te_eV, ne_cm3, ion):

    # get charge state distributions from ionization equilibrium for Ca
    atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])
    
    # fractial abundances
    logTe, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, 
                                                         #ne_tau=0.1e19, 
                                                         plot=False)

    fig = plt.figure(figsize=(10,7))
    ax = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
    a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 2, fig=fig) 
    a_legend.axis('off')
    
    ymin = np.inf
    ymax= -np.inf
    
    cols = iter(plt.cm.rainbow(np.linspace(0,1,len(pec_info.keys()))))
    #cols = aurora.get_color_cycle()

    for pp, (cs, pec_file) in enumerate(pec_info.items()):
        
        fz_3 = [1.,1.,1.] if fz is None else [fz[0,cs-1],fz[0,cs],fz[0,cs+1]]

        # load single PEC file from given li
        out = aurora.get_local_spectrum(pec_file, ion, ne_cm3, Te_eV, n0_cm3=0.0,
                                        ion_exc_rec_dens=[fz_3[0], fz_3[1], fz_3[2]],
                                        dlam_A = 0.0, plot_spec_tot=False, no_leg=True, plot_all_lines=True, ax=ax)



def load_solps_cdens():
    
    # load DIII-D SOLPS case with C
    SOLPS_path = '/fusion/projects/toolbox/sciortinof/SOLPS/Lmode_180533_2300/'
    SOLPS_run = 'test_shift_out_Csput_Y4pc_allwall'
    SOLPS_geqdsk = '/fusion/projects/toolbox/sciortinof/SOLPS/Lmode_180533_2300/baserun/180533_2300_efit05.gfile'


    # Order of ions in b2fstate['na'] is [D0, D+1, C0, C+1, C+2, …], see so.b2_species.
    # Recall that if EIRENE is being used the neutral (D0, C0, ...) densities in b2fstate are nonsense; neutral results are in fort44 and fort46.
    so = aurora.solps_case(SOLPS_path, SOLPS_geqdsk, SOLPS_run)

    so.plot2d_b2(so.quants['nn'], scale='log', label=so.labels['nn'])
    so.geqdsk.plot(only2D=True)

    # C3+ density
    so.plot2d_b2(so.b2fstate['na'][5], scale='log', label=r'$n_{C5+}$ [$m^{-3}$]')
    so.geqdsk.plot(only2D=True)

    rhop_fsa, nn_fsa, rhop_LFS, nn_LFS, rhop_HFS, nn_HFS = so.get_radial_prof(so.quants['nn'], plot=True)
    plt.gca().set_ylabel(r'$n_n$ [$m^{-3}$]')
    plt.gca().grid(True)
    plt.tight_layout()
    rhop_fsa, ne_fsa, rhop_LFS, ne_LFS, rhop_HFS, ne_HFS = so.get_radial_prof(so.quants['ne'], plot=False)

    fig, ax = plt.subplots()
    ax.semilogy(rhop_fsa, nn_fsa / ne_fsa, label='FSA')
    ax.semilogy(rhop_LFS, nn_LFS / ne_LFS, label='LFS midplane')
    ax.semilogy(rhop_HFS, nn_HFS / ne_HFS, label='HFS midplane')
    ax.grid('on')
    ax.legend(loc='best').set_draggable(True)
    ax.set_xlabel(r'$\rho_p$')
    ax.set_ylabel(r'$n_n/n_e$')
    plt.tight_layout()

if __name__=='__main__':
 
    ion = 'C' #'Al' #'F' # 'Ca'
    if ion=='Ca':
        base = '/home/sciortino/atomlib/atomdat_master/adf15/ca/'
    elif ion=='F':
        base = '/home/sciortino/atomlib/atomdat_master/adf15/f/'
    elif ion=='Al':
        base = '/home/sciortino/atomlib/atomdat_master/adf15/al/'
    elif ion=='C':
        base = '/home/sciortino/atomlib/atomdat_master/adf15/c/'        
    else:
        raise ValueError(f'point to directory for ADF15 files for ion {ion}')
        
    #out = aurora.read_adf15(base+'fs#ca15_10A_70A.dat')

    Te_eV=200 #100#1500
    ne_cm3=5e13

    # get charge state distributions from ionization equilibrium for Ca
    atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])
    
    # fractial abundances
    logTe, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, 
                                                         #ne_tau=0.1e19, 
                                                         plot=False)

    if ion=='Ca':
        cs_range = np.arange(10,20)
    elif ion=='F':
        cs_range = np.arange(4,9)
    elif ion=='Al':
        cs_range = np.arange(4,13)
    elif ion=='C':
        cs_range = [0,1,2,3,4,5]
    else:
        raise ValueError('specify charge state range')

    pec_files = []
    mult = []
    for Z in cs_range:
        #pec_files += [base+f'fs#ca{Z}_10A_70A.dat',]
        #pec_files += [base+f'fs#80A_1250A#{ion.lower()}{Z}.dat',]
        #pec_files += [base+f'fs#1200A_1250A#{ion.lower()}{Z}.dat',]

        path = aurora.get_adas_file_loc(f'pec96#c_pjr#c{Z}.dat', filetype='adf15')
        pec_files += [path,]
        
        mult += [fz[0,Z],]

    # add H lya
    filename = 'pec96#h_pju#h0.dat' # for D Ly-alpha
    path = aurora.get_adas_file_loc(filename, filetype='adf15')
    pec_files += [path,]
    mult += [1.,]
    
    #aurora.adf15_line_identification(pec_files, Te_eV=Te_eV, ne_cm3=ne_cm3, mult=mult)
        

    # plot with broadening:
    pec_info = {}
    for pec_file in pec_files:
        species = pec_file.split('#')[-1].split('.dat')[0].split('_')[0]
        cs = int(re.split('(\d+)', species)[1])
        pec_info[cs] = pec_file

    overplot_spectra(pec_info, Te_eV, ne_cm3, ion)


    #cdens = load_solps_cdens()
