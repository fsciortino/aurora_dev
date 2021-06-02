import aurora
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
import re

def overplot_spectra(pec_info, Te_eV, ne_cm3, ion):

    # get charge state distributions from ionization equilibrium for Ca
    atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])
    
    # fractial abundances
    logTe, fz, rates = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, 
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



if __name__=='__main__':

    base = '/home/sciortino/atomlib/atomdat_master/adf15/ca/'
    #out = aurora.read_adf15(base+'fs#ca15_10A_70A.dat')

    Te_eV=300#1500
    ne_cm3=5e13

    ion = 'Ca'

    # get charge state distributions from ionization equilibrium for Ca
    atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])
    
    # fractial abundances
    logTe, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, 
                                                         #ne_tau=0.1e19, 
                                                         plot=False)

    pec_files = []
    mult = []
    for Z in np.arange(10,20):
        pec_files += [base+f'fs#ca{Z}_10A_70A.dat',]
        mult += [fz[0,Z],]

    aurora.adf15_line_identification(pec_files, lines={'test':20}, Te_eV=Te_eV, ne_cm3=ne_cm3, mult=mult)
        

    # plot with broadening:
    pec_info = {}
    for pec_file in pec_files:
        species = pec_file.split('#')[-1].split('.dat')[0].split('_')[0]
        cs = int(re.split('(\d+)', species)[1])
        pec_info[cs] = pec_file

    #overplot_spectra(pec_info, Te_eV, ne_cm3, ion)
