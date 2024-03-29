'''Script to demonstrate production of synthetic spectra for the He-like Ar spectrum
using Aurora and atomic data originally compiled in the atomDB database by A. Foster.

NB: this is not ADAS data.

See application of this spectral model for the case of Ca ions in
F. Sciortino et al 2021 Nucl. Fusion 61 126060
https://iopscience.iop.org/article/10.1088/1741-4326/ac32f2 

Note: this script creates a single local synthetic spectrum. Doing such calculations
in an iterative framework using line-integrated data requires significant numerical optimization. 
Please write to francesco.sciortino@ipp.mpg.de for help on setting this up.

F.Sciortino, 12/14/2021

'''
import numpy as np, sys, os
import matplotlib.pyplot as plt
from scipy.constants import e as q_electron,k as k_B, h, m_p, c as c_speed
plt.ion()
import periodictable
from scipy.interpolate import interp1d
import aurora

from IPython import embed

home = os.path.expanduser('~')

import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 1.5 #3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16


ion = 'Ar' #'Ca' #'Ar'

if ion=='Ca':
    Z_ion=20
    filepath_h = home+'/atomlib/atomdat_master/atomdb/pec#ca19.dat'
    filepath_he = home+'/atomlib/atomdat_master/atomdb/pec#ca18.dat'
    filepath_li = home+'/atomlib/atomdat_master/atomdb/pec#ca17.dat'

elif ion=='Ar':
    Z_ion=18
    filepath_h = home+'/atomlib/atomdat_master/atomdb/pec#ar17.dat'
    filepath_he = home+'/atomlib/atomdat_master/atomdb/pec#ar16.dat'
    filepath_li = home+'/atomlib/atomdat_master/atomdb/pec#ar15.dat'
    
else:
    raise ValueError('Specify PEC files for this ion!')

Te_eV = 3.5e3
ne_cm3 = 1e14

fig = plt.figure()
fig.set_size_inches(9,6, forward=True)
ax1 = plt.subplot2grid((10,1),(0,0),rowspan = 1, colspan = 1, fig=fig)
ax2 = plt.subplot2grid((10,1),(1,0),rowspan = 9, colspan = 1, fig=fig, sharex=ax1)

# Find fractional abundances
atom_data = aurora.get_atom_data(ion,['scd','acd'])

# always include charge exchange, although n0_cm3 may be 0
logTe, fz = aurora.get_frac_abundances(atom_data, np.array([ne_cm3,]), np.array([Te_eV,]), plot=False)

dlam_A = 0.00015

# now add spectra
out = aurora.get_local_spectrum(filepath_li, ion, ne_cm3, Te_eV, n0_cm3=0.0,
                                ion_exc_rec_dens=[fz[0,-5], fz[0,-4], fz[0,-3]], # Be-like, Li-like, He-like
                                dlam_A = dlam_A, plot_spec_tot=False, no_leg=True, plot_all_lines=True, ax=ax2)
wave_final_li, spec_ion_li, spec_exc_li, spec_rec_li, spec_dr_li, spec_cx_li, ax = out

out= aurora.get_local_spectrum(filepath_he, ion, ne_cm3, Te_eV, n0_cm3=0.0,
                               ion_exc_rec_dens=[fz[0,-4], fz[0,-3], fz[0,-2]], # Li-like, He-like, H-like
                               dlam_A = dlam_A, plot_spec_tot=False, no_leg=True, plot_all_lines=True, ax=ax2)
wave_final_he, spec_ion_he, spec_exc_he, spec_rec_he, spec_dr_he, spec_cx_he, ax = out

out = aurora.get_local_spectrum(filepath_h, ion, ne_cm3, Te_eV, n0_cm3=0.0,
                                ion_exc_rec_dens=[fz[0,-3], fz[0,-2], fz[0,-1]], # He-like, H-like, fully stripped
                                dlam_A = dlam_A, plot_spec_tot=False, no_leg=True, plot_all_lines=True, ax=ax2)
wave_final_h, spec_ion_h, spec_exc_h, spec_rec_h, spec_dr_h, spec_cx_h, ax = out

spec_tot_li = spec_ion_li + spec_exc_li + spec_rec_li + spec_dr_li + spec_cx_li
spec_tot_he = spec_ion_he + spec_exc_he + spec_rec_he + spec_dr_he + spec_cx_he
spec_tot_h = spec_ion_h + spec_exc_h + spec_rec_h + spec_dr_h + spec_cx_h

# add plot of total spectrum
wave_all = np.linspace(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 10000) #A
spec_all = interp1d(wave_final_li, spec_tot_li, bounds_error=False, fill_value=0.0)(wave_all)
spec_all += interp1d(wave_final_he, spec_tot_he, bounds_error=False, fill_value=0.0)(wave_all)
spec_all += interp1d(wave_final_h, spec_tot_h, bounds_error=False, fill_value=0.0)(wave_all)
plt.gca().plot(wave_all, spec_all, 'k', label='total')



ax2.plot([], [], c='r', ls='--', label='ionization')
ax2.plot([], [], c='b', ls='--', label='excitation')
ax2.plot([], [], c='g', ls='--', label='radiative recomb')
ax2.plot([], [], c='m', ls='--', label='dielectronic recomb')
#ax2.plot([], [], c='c', ls='--', label='CX recomb')
#plt.gca().legend(loc='best').set_draggable(True)


with open(home+'/bsfc/data/hirexsr_wavelengths.csv', 'r') as f:
    lineData = [s.strip().split(',') for s in f.readlines()]
    lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
    lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
    lineName = np.array([ld[3] for ld in lineData[2:]])

# select only lines from Ca
xics_lams = lineLam[lineZ==Z_ion]
xics_names = lineName[lineZ==Z_ion]

for ii,_line in enumerate(xics_lams):
    if _line>ax2.get_xlim()[0] and _line<ax2.get_xlim()[1]:
        ax2.axvline(_line, c='r', ls='--')
        ax1.axvline(_line, c='r', ls='--')

        ax1.text(_line, 0.5, xics_names[ii], rotation=90, fontdict={'fontsize':14})
ax1.axis('off')


