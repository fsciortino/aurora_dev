import numpy as np
import matplotlib.pyplot as plt
import sys
from colradpy import colradpy
from scipy import constants
plt.ion()

n0 = np.zeros(30)
n0[0] = 1.
td_s = np.copy(n0)
td_t = np.geomspace(1.e-6,1,10000)
    
cv_te = np.array([20])
cv_ne = np.array([1.e12])

filepath = '/home/sciortino/adf04_files/c/c_adf04_adas/mom97_ls#c2.dat'
cv = colradpy(filepath,np.array([0]),cv_te,cv_ne,use_recombination=False,
              use_recombination_three_body=False)#,td_source=n0)

cv.solve_cr()


cv.get_nist_levels_txt()
cv.split_structure_terms_to_levels()
cv.shift_j_res_energy_to_nist()
cv.split_pec_multiplet()
wave_air = cv.data['processed']['split']['wave_air']
cv.make_dopp_broad(T=20 , m=1.673e-27*12,split_wave=wave_air)
cv.make_nat_broad(split_wave=wave_air)

doppler = cv.data['processed']['broadening']['dopp']
nat = cv.data['processed']['broadening']['nat']

aa = np.where( (wave_air >464.7) & (wave_air<465.2))[0]
#aa = np.arange(len(wave_air))

# Set threshold to PEC values (phot*cm^3/s)
pec_threshold=1e-30  #1e-31


#######  Visualize Doppler broadening #######

plt.figure()
for i in range(len(aa)):
    pec = cv.data['processed']['split']['pecs'][aa[i],0,0,0]
    prof = doppler['theta'][aa[i],:]
    if pec<pec_threshold:
        continue
    
    l = plt.plot(doppler['wave_arr'][aa[i],:],
             prof*pec,
             label='T$_i$=20 eV')
    
    plt.vlines(wave_air[aa[i]],
               pec*np.min(prof), pec*np.max(prof),
               colors=l[0].get_color(), ls='--')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (AU)')


#######  Visualize natural broadening #######

plt.figure()
for i in range(0,len(aa)):
    pec = cv.data['processed']['split']['pecs'][aa[i],0,0,0]
    prof = nat['theta'][aa[i],:]
    if pec<pec_threshold:
        continue
    
    l= plt.plot(nat['wave_arr'][aa[i],:],
                prof*pec,
                label='T$_i$=20 eV')
    
    plt.vlines(wave_air[aa[i]],
               pec*np.min(prof), pec*np.max(prof),
               colors=l[0].get_color())
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (AU)')


