import aurora
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from scipy.interpolate import interp1d



S_He = aurora.atomic.get_atom_data('He',['scd',])
S_C = aurora.atomic.get_atom_data('C',['scd',])
S_F = aurora.atomic.get_atom_data('F',['scd',])






fig, ax = plt.subplots(4,5, sharex=True, sharey=True)
scd_old = aurora.adas_file('/home/sciortino/atomlib/atomdat_master/newdat/scd85_ca.dat')
