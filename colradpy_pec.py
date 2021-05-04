'''
This script shows some of the functionality of ColRadPy and makes it clear how this could be useful for Aurora.
sciortino, 10/31/20
'''
import matplotlib.pyplot as plt
import numpy as np, sys, os
home = os.path.expanduser('~')
if home+'/ColRadPy' not in sys.path:
    sys.path.append(home+'/ColRadPy')
from colradpy import colradpy
plt.ion()
from matplotlib import cm

Te_grid = np.array([100])
ne_grid = np.array([1.e13]) #,1.e14,1.e15])


filepath = home+'/adf04_files/ca/ca_adf04_adas/'

files = {'ca8': 'mglike_lfm14#ca8.dat',
         'ca9': 'nalike_lgy09#ca9.dat',
         'ca10': 'nelike_lgy09#ca10.dat',
         'ca11': 'flike_mcw06#ca11.dat',
         #'ca14': 'clike_jm19#ca14.dat',  # unknown source; issue with format?
         'ca15': 'blike_lgy12#ca15.dat',
         'ca16': 'belike_lfm14#ca16.dat',
         'ca17': 'lilike_lgy10#ca17.dat',
         'ca18': 'helike_adw05#ca18.dat',  # Whiteford, R-matrix 2005: https://open.adas.ac.uk/detail/adf04/copaw][he/helike_adw05][ca18.dat
         #'ca19': 'copha#h_bn#97ca.dat', # O'Mullane, 2015: https://open.adas.ac.uk/detail/adf04/copha][h/copha][h_bn][97ca.dat
         }

colors = cm.rainbow(np.linspace(0, 1, len(files)))
fig, ax = plt.subplots()

res = {}
for ii,cs in enumerate(files.keys()):
    res[cs] = colradpy(filepath+files[cs],[0],Te_grid,ne_grid,use_recombination=False,  # use_rec has issue with ca8 file
                       use_recombination_three_body=True, temp_dens_pair=True)
    
    res[cs].make_ioniz_from_reduced_ionizrates()
    res[cs].suppliment_with_ecip()
    res[cs].make_electron_excitation_rates()
    res[cs].populate_cr_matrix()
    res[cs].solve_quasi_static()

    # plot lines
    ax.vlines(res[cs].data['processed']['wave_vac'],np.zeros_like(res[cs].data['processed']['wave_vac']),
              res[cs].data['processed']['pecs'][:,0,0],label=cs, color=colors[ii])
    

ax.set_xlim(0,200)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('PEC (ph cm$^3$ s$^{-1}$)')
ax.legend().set_draggable(True)


## Plot only lines with sufficiently high PEC (most likely to be measured)

pec_threshold=1e-20  #1e-31

fig, ax = plt.subplots()
for ii,cs in enumerate(files.keys()):
    idxs = np.where(res[cs].data['processed']['pecs'][:,0,0]>pec_threshold)[0]

    ax.vlines(res[cs].data['processed']['wave_vac'][idxs],
              np.zeros_like(res[cs].data['processed']['wave_vac'][idxs]),
              res[cs].data['processed']['pecs'][idxs,0,0],label=cs, color=colors[ii])

ax.set_xlim(0,200)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('PEC (ph cm$^3$ s$^{-1}$)')
ax.legend().set_draggable(True)


#plt.show(block=True)
