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
import aurora

Te_grid = np.array([1000])
ne_grid = np.array([1.e14]) #,1.e14,1.e15])


filepath = home+'/adf04_files/ca/ca_adf04_adas/'

files = {#'ca8': 'mglike_lfm14#ca8.dat',
         #'ca9': 'nalike_lgy09#ca9.dat',
         #'ca10': 'nelike_lgy09#ca10.dat',
         #'ca11': 'flike_mcw06#ca11.dat',
         #'ca14': 'clike_jm19#ca14.dat',  # unknown source; issue with format?
         #'ca15': 'blike_lgy12#ca15.dat',
         #'ca16': 'belike_lfm14#ca16.dat',
         #'ca17': 'lilike_lgy10#ca17.dat',
         'ca18': 'helike_adw05#ca18.dat',  # Whiteford, R-matrix 2005: https://open.adas.ac.uk/detail/adf04/copaw][he/helike_adw05][ca18.dat
         #'ca19': 'copha#h_bn#97ca.dat', # O'Mullane, 2015: https://open.adas.ac.uk/detail/adf04/copha][h/copha][h_bn][97ca.dat
         }

#colors = cm.rainbow(np.linspace(0, 1, len(files)))
#fig, ax = plt.subplots()

res = {}
for ii,cs in enumerate(files.keys()):
    res[cs] = colradpy(filepath+files[cs],[0],Te_grid,ne_grid,use_recombination=False,  # use_rec has issue with ca8 file
                       use_recombination_three_body=True, temp_dens_pair=True)
    
    res[cs].make_ioniz_from_reduced_ionizrates()
    res[cs].suppliment_with_ecip()
    res[cs].make_electron_excitation_rates()
    res[cs].populate_cr_matrix()
    res[cs].solve_quasi_static()


print(res['ca18'].data['processed']['acd'])

adas_data = aurora.get_atom_data('Ca',filetypes=['acd','scd'])
