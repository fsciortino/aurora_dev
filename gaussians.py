import matplotlib.pyplot as plt
plt.ion()
import numpy as np

import scipy.stats as stats
#plt.style.use('/home/sciortino/SPARC/sparc_plots.mplstyle')

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
#mpl.rcParams['xtick.labelsize'] = 16

mu=0
variance=1

sigma = np.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu+5*sigma, 1000)

plt.figure()
plt.plot(x, stats.norm.pdf(x,mu,sigma),c='b')
plt.xlabel(r'$x/\sigma$',fontsize=18)
plt.tight_layout()


plt.figure()
plt.plot(x, stats.norm.pdf(x,mu,sigma), c='b')
plt.plot(x, stats.norm.pdf(x,mu+1,sigma), c='r')
plt.xlabel(r'$x/\sigma$',fontsize=18)
plt.tight_layout()

plt.figure()
plt.plot(x, stats.norm.pdf(x,mu,sigma), c='b')
#plt.plot(x, stats.norm.pdf(x,mu+1,sigma), c='r')
plt.plot(x, stats.norm.pdf(x,mu,sigma+0.5), c='g')
plt.xlabel(r'$x/\sigma$',fontsize=18)
plt.tight_layout()


from numpy.polynomial.hermite import hermval
from scipy.integrate import simps

fig,ax = plt.subplots()
xx = np.linspace(-5, 5, 1000)
nvals = np.zeros(6)
for n in np.arange(len(nvals)):
    nn = np.zeros_like(nvals)
    nn[n] = 1.0
    ax.plot(xx, hermval(xx,nn), ls='-.', label=fr'n={n}')

ax.legend(loc='best', fontsize=16).set_draggable(True)
ax.grid(True)
ax.set_ylim([-10,20])
ax.set_xlim([-3,6])
ax.plot([0,0],ax.get_ylim(),'k-')
ax.plot(ax.get_xlim(),[0,0],'k-')
ax.set_ylabel(r'$He_n(x)$',fontsize=18)
ax.set_xlabel(r'$x$',fontsize=18)
plt.tight_layout()





###### Gauss-Hermite functions #######
fig_description='Gauss Hermite functions'
fig_source='Sciortino et al. - RSI 2021'
comment=' '
user_fullname='Francesco Sciortino'
from h5_data import h5_data
hdf_file=h5_data(f'hdf5_files/gauss_hermite_functions.hdf5',
                          fig_description=fig_description,
                          fig_source=fig_source,
                          comment=comment,
                          user_fullname = user_fullname)

fig,ax = plt.subplots()
xx = np.linspace(-5, 5, 1000)
nvals = np.zeros(5)
for n in np.arange(len(nvals)):
    nn = np.zeros_like(nvals)
    nn[n] = 1.0
    #norm = np.max(hermval(xx,nn)*np.exp(-xx**2/2)) #simps(hermval(xx,nn)*np.exp(-xx**2/2), xx)
    norm = (2.**n *np.math.factorial(n)*np.sqrt(np.pi))**(-0.5)
    ax.plot(xx, hermval(xx,nn)*np.exp(-xx**2/2)*norm, ls='-.', label=f'n={n}')

    hdf_file.add_dataset(f'Gauss-Hermite function, n={n}',legend=None,
                                  plot_info=f'n={n}',
                                  x_data=xx, x_units='', x_label='x',
                                  y_data=hermval(xx,nn)*np.exp(-xx**2/2)*norm,
                                  y_units='', y_label=r'$\phi_n(x)$')
    
ax.legend(loc='best', fontsize=16).set_draggable(True)
ax.grid(True)
ax.plot([0,0],ax.get_ylim(),'k-')
ax.plot(ax.get_xlim(),[0,0],'k-')
ax.set_ylabel(r'$\phi_n(x)$',fontsize=18)
ax.set_xlabel(r'$x$',fontsize=18)
plt.tight_layout()

