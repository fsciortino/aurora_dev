'''
Calculate SXR radiation terms for a specific filter function, creating iso-nuclear results similarly
to the ADAS ADF11 files (PLS and PRS). 

Convention:
filt_num_dict = {'125 um be': '14', '12 um be': '15', '140um be': '14', 'none': '0', 'axuv': '10'}

sciortino, 2020
'''
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import numpy as np
from colradpy import colradpy
plt.ion()
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.constants import h, c, e
import aurora

# read filter function
phot_energy=[]
transmission=[]
with open('/home/sciortino/aurora_dev/Be_filter_140um.txt','r') as f:
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
    
# Use gfile and statefile in local directory:
geqdsk = omfit_eqdsk.OMFITgeqdsk('/home/sciortino/Aurora/examples/example.gfile')
inputgacode = omfit_gapy.OMFITgacode('/home/sciortino/Aurora/examples/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid

#rhop = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
ne_cm3 = [1e14,] #inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
Te_eV = [1000,] #inputgacode['Te']*1e3  # keV --> eV

filepath = '/home/sciortino/adf04_files/c/c_adf04_adas/'

files = {'c0': 'mom97_ls#c0.dat',
         'c1': 'mom97_ls#c1.dat',
         'c2': 'mom97_ls#c2.dat',
         'c3': 'mom97_ls#c3.dat',
         'c4': 'mom97_ls#c4.dat',
         'c5': 'mom97_n#c5.dat'}

# filepath = '/home/sciortino/adf04_files/ca/ca_adf04_adas/'

# files = {'ca8': 'mglike_lfm14#ca8.dat',
#          'ca9': 'nalike_lgy09#ca9.dat',
#          'ca10': 'nelike_lgy09#ca10.dat',
#          'ca11': 'flike_mcw06#ca11.dat',
#          'ca14': 'clike_jm19#ca14.dat',
#          'ca15': 'blike_lgy12#ca15.dat',
#          'ca16': 'belike_lfm14#ca16.dat',
#          'ca17': 'lilike_lgy10#ca17.dat',
#          'ca18': 'helike_adw05#ca18.dat'}

colors = cm.rainbow(np.linspace(0, 1, len(files)))


res = {}
pls = {}
prs = {}
for ii,cs in enumerate(files.keys()):
    res[cs] = colradpy(filepath+files[cs],[0],Te_eV,ne_cm3,use_recombination=True,
                       use_recombination_three_body=True, temp_dens_pair=True)
    
    res[cs].make_ioniz_from_reduced_ionizrates()
    res[cs].suppliment_with_ecip()
    res[cs].make_electron_excitation_rates()
    res[cs].populate_cr_matrix()
    res[cs].solve_quasi_static()

    # Convolution with SXR filter:
    lam = res[cs].data['processed']['wave_vac']
    E_eV = h*c/(lam*1e-9) /e
    pec_exc = res[cs].data['processed']['pecs'][:,0]  # ph cm$^{-3}$ s$^{-1}$
    pec_recomb = res[cs].data['processed']['pecs'][:,1]  # ph cm$^{-3}$ s$^{-1}$

    trans = interp1d(phot_energy, transmission, kind='linear')(E_eV)

    pls[cs] = np.sum(pec_exc * E_eV * e * ne_cm3 * trans)
    prs[cs] = np.sum(pec_recomb * E_eV * e * ne_cm3 * trans)


# now read pls file using Aurora
atom_data = aurora.atomic.get_atom_data('C', ['pls'])
pls_adas = aurora.atomic.interp_atom_prof(atom_data['pls'],np.log10(ne_cm3), np.log10(Te_eV))
atom_data = aurora.atomic.get_atom_data('C', ['prs'])
prs_adas = aurora.atomic.interp_atom_prof(atom_data['prs'],np.log10(ne_cm3), np.log10(Te_eV))


print(r'PLS colradpy [ph cm$^{-3}$ s$^{-1}$]: ')
print(pls)

print(r'PLS ADAS [ph cm$^{-3}$ s$^{-1}$]: ')
print(pls_adas)

print(r'PRS colradpy [ph cm$^{-3}$ s$^{-1}$]: ')
print(prs)

print(r'PRS ADAS [ph cm$^{-3}$ s$^{-1}$]: ')
print(prs_adas)
