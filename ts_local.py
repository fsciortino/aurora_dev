import profiletools
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from plot_cmod_machine import overplot_machine
from omfit_classes import omfit_eqdsk

# L-mode (new)
shot = 1100308004; solps_run='Attempt14'
path = f'/home/sciortino/SOLPS/full_CMOD_runs/Lmode_1100308004/'

# H-mode
#shot = 1100305023; solps_run='Attempt23' # H-mode
#path = '/home/sciortino/SOLPS/full_CMOD_runs/Hmode_1100305023'

# I-mode
#shot = 1080416025; solps_run='Attempt15N'
#path = '/home/sciortino/SOLPS/full_CMOD_runs/Imode_1080416025'
#path = f'/home/sciortino/SOLPS/RR_Imode_attempt{case_num}_old/Output/'


t0 = 0.9
t1 = 1.1
    
p_ne = profiletools.neETS(shot, t_min=t0, t_max=t1, abscissa='RZ')
p_ne.time_average()

p_Te = profiletools.TeETS(shot, t_min=t0, t_max=t1, abscissa='RZ')
p_Te.time_average()


p_ne.plot_data()
p_Te.plot_data()


# all Thomson data is at the R=0.69m vertical slice (small rounding needed to avoid floating point errors)
assert len(np.unique(p_ne.X[:,0]))==1 and np.round(np.unique(p_ne.X[:,0])[0],5)==0.69

# same measurement points for ne and Te?
#assert np.allclose(p_ne.X[:,1], p_Te.X[:,1])
# ANS: NO! Sometimes either ne or Te points seemed to have been filtered out

print('Z values of Te Thomson points (time-averaged): ')
print(p_Te.X[:,1])


geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(device='CMOD',shot=shot,
                                                  time=(t0+t1)/2.*1e3, SNAPfile='EFIT20',
                                                  fail_if_out_of_range=False,
                                                  time_diff_warning_threshold=20)

fig,ax = plt.subplots(figsize=(8,10))
overplot_machine(shot, ax)
geqdsk.plot(only2D=True, ax=ax, color='r')
ax.plot(p_ne.X[:,0], p_ne.X[:,1], '*')
ax.set_xlabel(r'$R$ $[m]$')
ax.set_ylabel(r'$Z$ $[m]$')

