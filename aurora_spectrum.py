import numpy as np, sys
import matplotlib.pyplot as plt
from scipy.constants import e as q_electron,k as k_B, h, m_p, c as c_speed
plt.ion()
import periodictable
from scipy.interpolate import interp1d
import aurora

from IPython import embed



filepath_19='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca19.dat'
filepath_18='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca18.dat'
filepath_17='/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca17.dat'

Te_eV = 1000. #3.5e3 + 300. # add 300 eV for C-Mod instrumental function
ne_cm3 = 0.6e14

fig = plt.figure()
fig.set_size_inches(10,7, forward=True)
ax1 = plt.subplot2grid((10,1),(0,0),rowspan = 1, colspan = 1, fig=fig)
ax2 = plt.subplot2grid((10,1),(1,0),rowspan = 9, colspan = 1, fig=fig, sharex=ax1)

ax2.set_xlim([3.17, 3.215]) # A, He-lke Ca spectrum
#ax2.set_xlim([3.0, 3.215]) # includes Ly-a

with open('/home/sciortino/usr/python3modules/bsfc/data/hirexsr_wavelengths.csv', 'r') as f:
    lineData = [s.strip().split(',') for s in f.readlines()]
    lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
    lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
    lineName = np.array([ld[3] for ld in lineData[2:]])

# select only lines from Ca
xics_lams = lineLam[lineZ==20]
xics_names = lineName[lineZ==20]

for ii,_line in enumerate(xics_lams):
    if _line>ax2.get_xlim()[0] and _line<ax2.get_xlim()[1]:
        ax2.axvline(_line, c='r', ls='--')
        ax1.axvline(_line, c='r', ls='--')

        ax1.text(_line, 0.5, xics_names[ii], rotation=90, fontdict={'fontsize':14}) #, transform=ax1.transAxes)
ax1.axis('off')


# Find fractional abundances
atom_data = aurora.get_atom_data('Ca',['scd','acd'])

# always include charge exchange, although n0_cm3 may be 0
logTe, fz, rates = aurora.get_frac_abundances(atom_data, np.array([ne_cm3,]), np.array([Te_eV,]), plot=False)


# now add spectra
wave_final_A_18, spec_tot_18, ax = aurora.plot_local_spectrum(
    filepath_18, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
    ion_exc_rec_dens=[fz[0][-4], fz[0][-3], fz[0][-2]], # Li-like, He-like, H-like
    ax=ax2,  plot_spec_tot=False)
wave_final_A_17, spec_tot_17, ax = aurora.plot_local_spectrum(
    filepath_17, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
    ion_exc_rec_dens=[fz[0][-5], fz[0][-4], fz[0][-3]], # Be-like, Li-like, He-like
    ax=ax2, plot_spec_tot=False, no_leg=True)
wave_final_A_19, spec_tot_19, ax = aurora.plot_local_spectrum(
    filepath_19, 'Ca', ne_cm3, Te_eV, n0_cm3=0.0,
    ion_exc_rec_dens=[fz[0][-3], fz[0][-2], fz[0][-1]], # He-like, H-like, fully stripped
    ax=ax2, plot_spec_tot=False, no_leg=True)

# add plot of total spectrum
wave_all_A = np.linspace(3.17,3.24, 10000)
spec_all = interp1d(wave_final_A_18, spec_tot_18, bounds_error=False, fill_value=0.0)(wave_all_A)
spec_all += interp1d(wave_final_A_17, spec_tot_17, bounds_error=False, fill_value=0.0)(wave_all_A)
spec_all += interp1d(wave_final_A_19, spec_tot_19, bounds_error=False, fill_value=0.0)(wave_all_A)
plt.gca().plot(wave_all_A, spec_all, 'k', label='total')
plt.gca().legend(loc='best').set_draggable(True)
