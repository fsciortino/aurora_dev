import aurora
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from scipy.interpolate import interp1d

# comparison of scd files
fig, ax = plt.subplots(4,5, sharex=True, sharey=True)
scd_old = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/newdat/scd85_ca.dat')
scd_old.plot(fig=fig, axes=ax)
scd_new = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/pue2020_data/scd50_ca.dat')
scd_new.plot(fig=fig, axes=ax)


# comparison of acd files
fig, ax = plt.subplots(4,5, sharex=True, sharey=True)
acd_old = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/newdat/acd85_ca.dat')
acd_old.plot(fig=fig, axes=ax)
acd_new = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/pue2020_data/acd89_ca.type_a_large')
acd_new.plot(fig=fig, axes=ax)



# Find ratios #
for cs in np.arange(acd_old.data.shape[0]):
    acd_old_interp = interp1d(acd_old.logT, acd_old.data[cs,:,0], bounds_error=False)(acd_new.logT)
    plt.figure()
    plt.plot(10**acd_new.logT, (10**acd_new.data[cs,:,0])/(10**acd_old_interp))
    plt.xlabel(r'$T_e$ [eV]')
    plt.ylabel(fr'ACD Ca{cs+1}+ ratio new/old')
    plt.xlim([0,5000])

# Now for SCD
for cs in np.arange(scd_old.data.shape[0]):
    scd_old_interp = interp1d(scd_old.logT, scd_old.data[cs,:,0], bounds_error=False)(scd_new.logT)
    plt.figure()
    plt.plot(10**scd_new.logT, (10**scd_new.data[cs,:,0])/(10**scd_old_interp))
    plt.xlabel(r'$T_e$ [eV]')
    plt.ylabel(fr'SCD Ca{cs}+ ratio new/old')
    plt.xlim([0,5000])
    plt.ylim([0.5, 2])



# ---------------- PLS and PRS ----------
# comparison of PLS files
fig, ax = plt.subplots(4,5, sharex=True, sharey=True)
pls_old = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/pue2020_data/pls_Ca_9.dat')
pls_old.plot(fig=fig, axes=ax)
pls_new = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/pue2021_data/pls_caic_mix_Ca_9.dat')
pls_new.plot(fig=fig, axes=ax)



# comparison of PRS files
fig, ax = plt.subplots(4,5, sharex=True, sharey=True)
prs_old = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/pue2020_data/prs_Ca_9.dat')
prs_old.plot(fig=fig, axes=ax)
prs_new = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/pue2021_data/prs_Ca_9.dat')
prs_new.plot(fig=fig, axes=ax)

