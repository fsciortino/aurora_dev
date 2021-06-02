import aurora
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from scipy.interpolate import interp1d
from omfit_classes import omfit_eqdsk, omfit_gapy

# load some kinetic profiles
examples_dir = '/home/sciortino/Aurora/examples'
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
profs = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = {'Te':{}, 'ne':{}}
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(profs['polflux']/profs['polflux'][-1])
kp['ne']['vals'] = profs['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = profs['Te']*1e3  # keV --> eV


####
imp = 'Ca'

atom_data = aurora.get_atom_data(imp)
R_rates = aurora.interp_atom_prof(atom_data['acd'],
                                  np.log10(kp['ne']['vals']), np.log10(kp['Te']['vals']),
                                  x_multiply=True)

fig,ax = plt.subplots()
ls_cycle = aurora.get_ls_cycle()
for cs in np.arange(1,10): #R_rates.shape[1]):
    lss = next(ls_cycle)
    zz = R_rates.shape[1]-cs+1
    ax.semilogy(kp['ne']['rhop'], R_rates[:,-cs], lss, label=f'{imp}{zz}+')
ax.legend(loc='best').set_draggable(True)
ax.set_xlabel(r'$\rho_p$')
ax.set_ylabel(r'R')

########

imp = 'Al'

atom_data = aurora.get_atom_data(imp)
R_rates = aurora.interp_atom_prof(atom_data['acd'],
                                  np.log10(kp['ne']['vals']), np.log10(kp['Te']['vals']),
                                  x_multiply=True)

fig,ax = plt.subplots()
ls_cycle = aurora.get_ls_cycle()
for cs in np.arange(1,10): #R_rates.shape[1]):
    lss = next(ls_cycle)
    zz = R_rates.shape[1]-cs+1
    ax.semilogy(kp['ne']['rhop'], R_rates[:,-cs], lss, label=f'{imp}{zz}+')
ax.legend(loc='best').set_draggable(True)
ax.set_xlabel(r'$\rho_p$')
ax.set_ylabel(r'R')
