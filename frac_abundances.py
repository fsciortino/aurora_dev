'''
Script to obtain fractional abundances from Aurora's reading of ADAS data.
Note that this is a calculation from atomic physics only, i.e. no transport is applied.

It is recommended to run this in IPython.
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_gapy
import scipy,sys,os
import time
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
import aurora

plot=True

ion='Fe' #'Ar' #'Ca' #'Cl' #'Mg' #'Al' #Mg'

# read in some kinetic profiles
examples_dir = '/home/sciortino/Aurora/examples'
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# transform rho_phi (=sqrt toroidal flux) into rho_psi (=sqrt poloidal flux) and save kinetic profiles
rhop = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
ne_vals = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
Te_vals = inputgacode['Te']*1e3  # keV --> eV

# get charge state distributions from ionization equilibrium for Ca
atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])

# get fractional abundances on ne (cm^-3) and Te (eV) grid
logTe, fz, rates = aurora.atomic.get_frac_abundances(
    atom_data, ne_vals, Te_vals, rho=rhop, plot=plot)

# compare to fractial abundances obtained with ne*tau=0.1e20 m^-3.s
logTe, fz, rates = aurora.atomic.get_frac_abundances(
    atom_data, ne_vals, Te_vals, rho=rhop, ne_tau=0.1e19,
    plot=plot, ax=plt.gca() if plot else None, ls='--')


#plt.gca().set_xlim([0.8,None])


# find temperature of max He-like fractional abundance for many ions
from omfit_commonclasses.utils_math import atomic_element

# scan Te at density of 1e20 m^-3
Te_vals = np.linspace(1, 5e3, 10000)
ne_vals = 0.5e20 * np.ones_like(Te_vals)
ZZ = []
Te_Helike_max = []
for Z in np.arange(30):
    try:
        out = atomic_element(Z=Z)
        ion = out[list(out.keys())[0]]['symbol']
        atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])

        logTe, fz, rates = aurora.atomic.get_frac_abundances(
            atom_data, ne_vals, Te_vals, plot=False)

        Te_Helike_max.append( 10**logTe[np.argmax(fz[:,-3])])
        ZZ.append(Z)

    except:
        pass

plt.figure();
plt.plot(ZZ, Te_Helike_max, 'ro-')
