import numpy as np
import os
from scipy.constants import h,c,e
import matplotlib.pyplot as plt
plt.ion()

loc = '/home/sciortino/ColRadPy/atomic/nist_energies'

ion_sel = 'W'

_E_eV = []
cs = []
for filename in os.listdir(loc):
    if not filename.startswith('#') and not filename.endswith('~'):
        ion = filename.split('_')[0]

        if ion!=ion_sel.lower():
            continue
        charge = int(filename.split('_')[1])

        # read energy from file
        with open(loc+os.sep+filename,'r') as f:
            cont = f.readlines()

        # last value is energy in cm^-1
        E_cm_val = float(cont[-1].split(',')[-1].strip('/n'))
        E_eV_val = h*c/e*(E_cm_val*100.)

        cs.append(charge)
        _E_eV.append(E_eV_val)

idx = np.argsort(cs)
E_eV = np.array(_E_eV)[idx]

I = []
for i in np.arange(1,len(E_eV)-1):
    I.append( 2*(E_eV[i+1] - E_eV[i])/(E_eV[i+1]+E_eV[i]))

plt.figure()
plt.plot(I)

I_mean = np.convolve(I, np.ones(7)/7, mode='same')
plt.plot(I_mean)

