import numpy as np, sys
import matplotlib.pyplot as plt
from scipy.constants import e as q_electron,k as k_B, h, m_p, c as c_speed
plt.ion()
import periodictable
from scipy.interpolate import interp1d
import aurora

from IPython import embed
plt.style.use('./plots.mplstyle')

ion = 'F' #'Ca' #'Ar'

filepaths = {}

if ion=='Ca':
    Z_ion=20

    filepaths[10] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca10_10A_70A.dat'
    filepaths[11] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca11_10A_70A.dat'
    filepaths[12] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca12_10A_70A.dat'
    filepaths[13] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca13_10A_70A.dat'
    filepaths[14] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca14_10A_70A.dat'
    filepaths[15] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca15_10A_70A.dat'
    filepaths[16] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca16_10A_70A.dat'
    filepaths[17] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca17_10A_70A.dat'
    filepaths[18] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca18_10A_70A.dat'  # lower quality
    filepaths[19] = '/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca19_10A_70A.dat'  # lower quality
    
elif ion=='Ar':
    Z_ion=18
    filepaths[19] = '/home/sciortino/atomlib/atomdat_master/atomdb/pec#ar17.dat'
    filepaths[18] = '/home/sciortino/atomlib/atomdat_master/atomdb/pec#ar16.dat'
    filepaths[17] = '/home/sciortino/atomlib/atomdat_master/atomdb/pec#ar15.dat'

elif ion=='F':
    Z_ion=9
    filepaths[7] = '/home/sciortino/atomlib/atomdat_master/adf15/f/fs#f7_10A_70A.dat'
    filepaths[8] = '/home/sciortino/atomlib/atomdat_master/adf15/f/fs#f8_10A_70A.dat'
    
else:
    raise ValueError('Specify PEC files for this ion!')

Te_eV = 1e3
Ti_eV = 1e5 # effective broadening
ne_cm3 = 1e14

fig = plt.figure()
fig.set_size_inches(10,7, forward=True)
ax1 = plt.subplot2grid((10,1),(0,0),rowspan = 1, colspan = 1, fig=fig)
ax2 = plt.subplot2grid((10,1),(1,0),rowspan = 9, colspan = 1, fig=fig, sharex=ax1)


# with open('/home/sciortino/usr/python3modules/bsfc/data/hirexsr_wavelengths.csv', 'r') as f:
#     lineData = [s.strip().split(',') for s in f.readlines()]
#     lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
#     lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
#     lineName = np.array([ld[3] for ld in lineData[2:]])

# # select only lines from Ca
# xics_lams = lineLam[lineZ==Z_ion]
# xics_names = lineName[lineZ==Z_ion]

# for ii,_line in enumerate(xics_lams):
#     if _line>ax2.get_xlim()[0] and _line<ax2.get_xlim()[1]:
#         ax2.axvline(_line, c='r', ls='--')
#         ax1.axvline(_line, c='r', ls='--')

#         ax1.text(_line, 0.5, xics_names[ii], rotation=90, fontdict={'fontsize':14})
ax1.axis('off')


# Find fractional abundances
atom_data = aurora.get_atom_data(ion,['scd','acd'])

# always include charge exchange, although n0_cm3 may be 0
logTe, fz = aurora.get_frac_abundances(atom_data, np.array([ne_cm3,]), np.array([Te_eV,]), plot=False)

wave_all = np.linspace(10, 70, 10000) #A
spec_all = np.zeros_like(wave_all)

# now add spectra
for cs in filepaths:
    out = aurora.get_local_spectrum(filepaths[cs], ion, ne_cm3, Te_eV, Ti_eV=Ti_eV, n0_cm3=0.0,
                                    ion_exc_rec_dens=[fz[0][cs-Z_ion-3], fz[0][cs-Z_ion-2], fz[0][cs-Z_ion-1]],
                                    plot=False)
    wave_final, spec_ion, spec_exc, spec_rr, spec_dr, spec_cx, ax = out
    
    ax2.plot(wave_final, spec_ion, c='r', label='' if cs!=list(filepaths.keys())[0] else 'ionization')
    ax2.plot(wave_final, spec_exc, c='b', label='' if cs!=list(filepaths.keys())[0] else 'excitation')
    ax2.plot(wave_final, spec_rr, c='g', label='' if cs!=list(filepaths.keys())[0] else 'radiative recomb')
    ax2.plot(wave_final, spec_dr, c='m', label='' if cs!=list(filepaths.keys())[0] else 'dielectronic recomb')
    #ax2.plot(wave_final, spec_cx, c='c', label='' if cs!=list(filepaths.keys())[0] else 'charge exchange recomb')

    spec_tot = spec_ion + spec_exc + spec_rr + spec_dr #+ spec_cx
    
    spec_all += interp1d(wave_final, spec_tot, bounds_error=False, fill_value=0.0)(wave_all)



ax2.plot(wave_all, spec_all, 'k', label='total')
ax2.legend(loc='best').set_draggable(True)
ax2.set_xlabel(r'$\lambda$ [$\AA$]')
ax2.set_ylabel(r'$\epsilon$ [A.U.]')
plt.tight_layout()
