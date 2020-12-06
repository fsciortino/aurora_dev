'''
Calculate SXR radiation terms for a specific filter function, creating iso-nuclear results similarly
to the ADAS ADF11 files (PLS and PRS). 

Convention:
filt_num_dict = {'125 um be': '14', '12 um be': '15', '140um be': '14', 'none': '0', 'axuv': '10'}
Note that PLS and PRS files used for DIII-D also have a 250nm SiO3 filter included.

sciortino, 2020
'''
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import numpy as np,os
from colradpy import colradpy
plt.ion()
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.constants import h, c, e
import aurora

# read filter function
phot_energy=[]
transmission=[]
with open('/home/sciortino/aurora_dev/Be_filter_125um.txt','r') as f:
    contents = f.readlines()
for line in contents[2:]:
    tmp = line.strip().split()
    phot_energy.append(float(tmp[0]))
    transmission.append(float(tmp[1]))
phot_energy=np.concatenate(([0.,], np.array(phot_energy)))
transmission=np.concatenate(([0.,],np.array(transmission)))

fig,ax = plt.subplots()
ax.plot(phot_energy, transmission)
ax.set_xlabel('Photon energy [eV]')
ax.set_ylabel('Transmission')
    
ne_cm3 = [1e14,] # cm^-3
Te_eV = [100,] # eV

imp = 'Ca'

if imp=='Be':
    filepath = '/home/sciortino/adf04_files/be/be_adf04_adas/'
elif imp=='C':
    filepath = '/home/sciortino/adf04_files/c/c_adf04_adas/'
elif imp=='N':
    filepath = '/home/sciortino/adf04_files/n/n_adf04_adas/'
elif imp=='Ca':
    filepath = '/home/sciortino/adf04_files/ca/ca_adf04_adas/'
else:
    raise ValueError('Unspecified ADF04 files path for this species!')

filenames = os.listdir(filepath)
# sort files based on their charge
filenames = np.array(filenames)[np.argsort([file.split('#')[-1].split('.')[0] for file in filenames])]

res = {}
pls = {}
prs = {}
for filename in filenames:
    cs = filename.split('#')[-1].split('.')[0]
    
    res[cs] = colradpy(filepath+filename,[0],Te_eV,ne_cm3,use_recombination=True,
                       use_recombination_three_body=True, temp_dens_pair=True)
    
    res[cs].make_ioniz_from_reduced_ionizrates()
    res[cs].suppliment_with_ecip()
    res[cs].make_electron_excitation_rates()
    res[cs].populate_cr_matrix()
    res[cs].solve_quasi_static()

    # Convolution with SXR filter:
    lam_nm = res[cs].data['processed']['wave_vac']
    E_J = h*c/(lam_nm*1e-9)
    E_eV = E_J/e
    
    # PEC units: photons*cm^3/s
    pec_exc = res[cs].data['processed']['pecs'][:,0]
    pec_recomb = res[cs].data['processed']['pecs'][:,1] 

    trans = interp1d(phot_energy, transmission, kind='linear')(E_eV)

    # obtain pls and prs in W*cm^3 by multiplying by energy in Joules and by transmission of filter
    pls[cs] = np.sum(pec_exc * E_J * trans)
    prs[cs] = np.sum(pec_recomb * E_J * trans)


# now read pls file using Aurora -- default: index 14, corresponding to DIII-D's 125 um Be filter (with thin SiO3 layer)
atom_data = aurora.atomic.get_atom_data(imp, ['pls'])
pls_adas = aurora.atomic.interp_atom_prof(atom_data['pls'],np.log10(ne_cm3), np.log10(Te_eV))
atom_data = aurora.atomic.get_atom_data(imp, ['prs'])
prs_adas = aurora.atomic.interp_atom_prof(atom_data['prs'],np.log10(ne_cm3), np.log10(Te_eV))


# For PLS units, see e.g. https://open.adas.ac.uk/detail/adf11/pls96/pls96_c.dat

# print PLS and PRS calculations from ColRadPy and directly from ADAS files:
print(' ')
print('--'*10)
print(fr'PLS colradpy [W cm^3]: {pls}')

print(fr'PLS ADAS [W cm^3]: {list(pls_adas[0])}')

print('    ')
print(fr'PRS colradpy [W cm^3]: {prs}')

print(fr'PRS ADAS [W cm^3]: {list(prs_adas[0])}')
