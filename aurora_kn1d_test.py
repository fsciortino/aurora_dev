import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import sys, os
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora


def get_midplane_D2_pressure(shot,time_s):
    from omfit_mds import OMFITmdsValue
    
    node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE', TDI='\\EDGE::TOP.GAS.RATIOMATIC.F_SIDE')

    data = node.data()
    time_vec = node.dim_of(0)

    # find average near chosen time
    indt_u = np.argmin(np.abs(time_vec- (time_s + 1e-3)))
    indt_d = np.argmin(np.abs(time_vec- (time_s - 1e-3)))
    inds = slice(indt_d, indt_u)
    return np.mean(data[indt_d:indt_u])

    
# Run KN1D to get atomic neutral density
p_H2_mTorr=get_midplane_D2_pressure(1080416025, 1.0)
print('p_H2: ', p_H2_mTorr)
