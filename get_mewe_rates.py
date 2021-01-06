import numpy as np
import matplotlib.pyplot as plt
from lines import *
plt.ion()
from cmod_atomic_data import get_atomic_data
import matplotlib as mpl
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18
mpl.rcParams['axes.labelsize']=18
mpl.rcParams['axes.titlesize']=18
mpl.rcParams['legend.fontsize']= 18
#mpl.rc('text',usetex=True)    # unc
import omfit_eqdsk, omfit_gapy
import aurora
from scipy.constants import e as q_electron, h, c
from scipy.interpolate import interp1d

def compute_Helike_rates(Z, ne, Te):
    """Compute the He-like photon emissivity coefficients, 
    separating them into ionizing, recombining and excitation components. 

    This routine uses the same algorithm as the IDL program lines.pro and lines.py.

    Parameters
    ----------
    Z : int
        The atomic number of the element to compute the lines for.
    ne : array, (`n_space`,) or (`n_time`, `n_space`)
        The electron densities, either a stationary profile, or profiles as a
        function of time. Units are cm^-3.
    Te : array, (`n_space`,) or (`n_time`, `n_space`)
        The electron temperatures, either a stationary profile, or profiles as a
        function of time. Units are keV.
    """
    atdata = get_atomic_data() #read_atdata()

    try:
        atdata = atdata[Z]
    except KeyError:
        raise ValueError("No atomic physics data for Z={Z:d}!".format(Z=Z))
    
    # Set up the return values:
    lam = [l.lam for l in atdata]
    E = [l.E for l in atdata]
    q = [l.q for l in atdata]
    comment = [l.comment for l in atdata]
    
    line_types = np.asarray([ld.data_type for ld in atdata])  
    He_like_lines, = np.where(line_types == 8)

    # Enforce the condition that the lines be in order in the file:
    He_like_lines.sort()

    if len(He_like_lines) > 0:
        S1 = SSS(Te, atdata[He_like_lines[0]].p[0:6])
        S2 = SSS(Te, atdata[He_like_lines[1]].p[0:6])
        S3 = SSS(Te, atdata[He_like_lines[2]].p[0:6])
        S4 = SSS(Te, atdata[He_like_lines[3]].p[0:6])
        S5 = SSS(Te, atdata[He_like_lines[4]].p[0:6])
        S6 = SSS(Te, atdata[He_like_lines[5]].p[0:6])

        SPR1 = S1 * atdata[He_like_lines[0]].p[6]
        SPR2 = S2 * atdata[He_like_lines[1]].p[6]
        SPR3 = S3 * atdata[He_like_lines[2]].p[6]
        SPR4 = S4 * ALPHAZ(Te, atdata[He_like_lines[3]].p[0], S2, S3, S6, S4)
        SPR5 = S5 * atdata[He_like_lines[4]].p[6]
        SPR6 = S6 * atdata[He_like_lines[5]].p[6]

        SMP1P = SSSDPR(
            Te,
            Z,
            atdata[He_like_lines[4]].p[0],
            atdata[He_like_lines[0]].p[0],
            atdata[He_like_lines[0]].p[7],
            atdata[He_like_lines[0]].p[8]
        )
        SM2 = SSSDPR(
            Te,
            Z,
            atdata[He_like_lines[3]].p[0],
            atdata[He_like_lines[1]].p[0],
            atdata[He_like_lines[1]].p[7],
            atdata[He_like_lines[1]].p[8]
        )
        SM1 = SSSDPR(
            Te,
            Z,
            atdata[He_like_lines[3]].p[0],
            atdata[He_like_lines[2]].p[0],
            atdata[He_like_lines[2]].p[7],
            atdata[He_like_lines[2]].p[8]
        )
        SM0 = SSSDPR(
            Te,
            Z,
            atdata[He_like_lines[3]].p[0],
            atdata[He_like_lines[5]].p[0],
            atdata[He_like_lines[5]].p[7],
            atdata[He_like_lines[5]].p[8]
        )

        S1PMP = SSSDPRO(
            Te,
            0.333,
            atdata[He_like_lines[4]].p[0],
            atdata[He_like_lines[0]].p[0],
            SMP1P
        )
        S2M = SSSDPRO(
            Te,
            0.6,
            atdata[He_like_lines[3]].p[0],
            atdata[He_like_lines[1]].p[0],
            SM2
        )
        S1M = SSSDPRO(
            Te,
            1.0,
            atdata[He_like_lines[3]].p[0],
            atdata[He_like_lines[2]].p[0],
            SM1
        )
        S0M = SSSDPRO(
            Te,
            3.0,
            atdata[He_like_lines[3]].p[0],
            atdata[He_like_lines[5]].p[0],
            SM0
        )

        SLIF = SSSLI(Te, atdata[He_like_lines[4]].p[9], 0.5)
        SLIZ = SSSLI(Te, atdata[He_like_lines[3]].p[9], 1.5)

        ALPHRRW = RADREC(Te, Z, atdata[He_like_lines[0]].p[10:16])
        ALPHRRX = RADREC(Te, Z, atdata[He_like_lines[1]].p[10:16])
        ALPHRRY = RADREC(Te, Z, atdata[He_like_lines[2]].p[10:16])
        ALPHRRZ = RADREC(Te, Z, atdata[He_like_lines[3]].p[10:16])
        ALPHRRF = RADREC(Te, Z, atdata[He_like_lines[4]].p[10:16])
        ALPHRRO = RADREC(Te, Z, atdata[He_like_lines[5]].p[10:16])

        T1DR = np.exp(-6.80 * (Z + 0.5)**2 / (1e3 * Te))
        T2DR = np.exp(-8.77 * Z**2 / (1e3 * Te))
        T3DR = np.exp(-10.2 * Z**2 / (1e3 * Te))
        T0DR = 5.17e-14 * Z**4 / (1e3 * Te)**1.5

        C1 = 12.0 / (1.0 + 6.0e-6 * Z**4)
        C2 = 18.0 / (1.0 + 3.0e-5 * Z**4)
        C3 = 69.0 / (1.0 + 5.0e-3 * Z**3)
        ALPHDRW = T0DR * (C1 * T1DR + C2 * T2DR + C3 * T3DR)

        C1 = 1.9
        C2 = 54.0 / (1.0 + 1.9e-4 * Z**4)
        C3 = (
            380.0 / (1.0 + 5.0e-3 * Z**3) * 2.0 * (Z - 1)**0.6 /
            (1e3 * Te)**0.3 / (1.0 + 2.0 * (Z - 1)**0.6 / (1e3 * Te)**0.3)
        )
        ALPHDRX = T0DR * 5.0 / 9.0 * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
        ALPHDRY = T0DR * 3.0 / 9.0 * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
        ALPHDRO = T0DR * 1.0 / 9.0 * (C1 * T1DR + C2 * T2DR + C3 * T3DR)

        C1 = 3.0 / (1.0 + 3.0e-6 * Z**4)
        C2 = 0.5 / (1.0 + 2.2e-5 * Z**4)
        C3 = 6.3 / (1.0 + 5.0e-3 * Z**3)
        ALPHDRF = T0DR * (C1 * T1DR + C2 * T2DR + C3 * T3DR)

        C1 = 9.0 / (1.0 + 7.0e-5 * Z**4)
        C2 = 27.0 / (1.0 + 8.0e-5 * Z**4)
        C3 = 380.0 / (1.0 + 5.0e-3 * Z**3) / (1.0 + 2.0 * (Z - 1)**0.6 / (1e3 * Te)**0.3)
        ALPHDRZ = T0DR * (C1 * T1DR + C2 * T2DR + C3 * T3DR)

        ALPHW = ALPHRRW + ALPHDRW
        ALPHX = ALPHRRX + ALPHDRX
        ALPHY = ALPHRRY + ALPHDRY
        ALPHZ = ALPHRRZ + ALPHDRZ
        ALPHF = ALPHRRF + ALPHDRF
        ALPHO = ALPHRRO + ALPHDRO

        # ----------------------------
        # Calculation for W line:
        #NA1 = (n_Li * SLIF + n_He * SPR5 + n_H * ALPHF) / (atdata[He_like_lines[4]].p[16] + ne * SMP1P)
        #NA2 = (n_He * SPR1 + n_H * ALPHW) / (ne * SMP1P)
        NA3 = (atdata[He_like_lines[0]].p[16] + ne * S1PMP) / (ne * SMP1P)
        NA4 = (atdata[He_like_lines[0]].p[17] + ne * S1PMP) / (atdata[He_like_lines[4]].p[16] + ne * SMP1P)
        #NW = atdata[He_like_lines[0]].p[16] * ne * (NA1 + NA2) / (NA3 - NA4)

        # Compute ionization, recombination and excitation components for W line
        C = atdata[He_like_lines[0]].p[16] * ne / (NA3 - NA4)
        NA1_xLi =  SLIF / (atdata[He_like_lines[4]].p[16] + ne * SMP1P)
        w_ioniz_comp = C* NA1_xLi 

        NA1_xHe = SPR5 / (atdata[He_like_lines[4]].p[16] + ne * SMP1P)
        NA2_xHe = SPR1 / (ne * SMP1P)
        w_exc_comp = C* (NA1_xHe+NA2_xHe)

        NA1_xH = ALPHF / (atdata[He_like_lines[4]].p[16] + ne * SMP1P)
        NA2_xH = ALPHW / (ne * SMP1P)
        w_recomb_comp = C* (NA1_xH + NA2_xH)
        
        w_comps = [w_ioniz_comp, w_exc_comp, w_recomb_comp]

        # ----------------------------
        # Calculation for Z line:
        NA1_xLi = SLIZ
        NA2_xHe = SPR4 + SPR6 + SPR3 / (
                1.0 + atdata[He_like_lines[2]].p[16] / (
                    atdata[He_like_lines[2]].p[17] + ne * S1M
                )
            )
        
        NA3_xHe =  SPR2 / (  1.0 + atdata[He_like_lines[1]].p[16] / (
            atdata[He_like_lines[1]].p[17] + ne * S2M  )
        )
        
        NA4_xH = ALPHZ + ALPHO + ALPHY / (
                1.0 + atdata[He_like_lines[2]].p[16] / (
                    atdata[He_like_lines[2]].p[17] + ne * S1M
                )
            )

        NA5_xH = ALPHX / (  1.0 + atdata[He_like_lines[1]].p[16] / (
                atdata[He_like_lines[1]].p[17] + ne * S2M
            ))
        
        NA6 = ne / atdata[He_like_lines[3]].p[16] * SM2 / (
            1.0 + (
                atdata[He_like_lines[1]].p[17] + ne * S2M
            ) / atdata[He_like_lines[1]].p[16]
        )
        NA7 = ne / atdata[He_like_lines[3]].p[16] * SM1 / (
            1.0 + (
                atdata[He_like_lines[2]].p[17] + ne * S1M
            ) / atdata[He_like_lines[2]].p[16]
        )
        #NZ = ne * (NA1 + NA2 + NA3 + NA4 + NA5) / (1.0 + NA6 + NA7)

        # Compute ionization, recombination and excitation components for Z line
        CC = ne / (1.0 + NA6 + NA7)

        z_ioniz_comp = CC * NA1_xLi
        z_exc_comp = CC * (NA2_xHe + NA3_xHe)
        z_recomb_comp = CC * (NA4_xH + NA5_xH)

        z_comps = [z_ioniz_comp, z_exc_comp, z_recomb_comp ]


        # cannot easily split x,y rates as for the w and z lines because they depend on z emissivity value!
        # -----------------------------------
        # Calculation for X line:
        #NA1 = n_He * SPR2 + n_H * ALPH
        NA1_xHe = SPR2
        NA1_xH = ALPHX

        NA2 = 1.0 + ( atdata[He_like_lines[1]].p[17] + ne * S2M  ) / atdata[He_like_lines[1]].p[16]
        NA3 = ne * SM2 / NA2
        #NX = ne * NA1 / NA2 + NA3 * NZ / atdata[He_like_lines[3]].p[16]
        NX_xHe = ne * NA1_xHe / NA2
        NX_xH = ne * NA1_xH / NA2
        NX_xZ = NA3 /  atdata[He_like_lines[3]].p[16]

        x_comps = [NX_xHe, NX_xH, NX_xZ ] #  different meaning than for W and Z lines!

        
        # Calculation for Y line:
        #NA1 = n_He * SPR3 + n_H * ALPHY
        NA1_xHe = SPR3
        NA1_xH = ALPHY
        NA2 = 1.0 + (atdata[He_like_lines[2]].p[17] + ne * S1M) / atdata[He_like_lines[2]].p[16]
        NA3 = ne * SM1 / NA2
        #NY = ne * NA1 / NA2 + NA3 * NZ / atdata[He_like_lines[3]].p[16]
        #em[:, He_like_lines[2], :] = NY
        NY_xHe = ne * NA1_xHe/NA2
        NY_xH = ne * NA1_xH/NA2
        NY_xZ = NA3 / atdata[He_like_lines[3]].p[16]

        y_comps = [NY_xHe, NY_xH, NY_xZ] # different meaning than for W and Z lines!


        return w_comps, z_comps, x_comps, y_comps



def Helike_emiss_metrics(imp='Ca', cs_den=None, rhop=None,
                         plot_individual_contributions=False, axs = None):
    ''' Obtain R(Te) and G(ne) from ratios of w,z,x,y He-like lines for an ion 
    '''
    
    # Use gfile and statefile in local directory:
    geqdsk = omfit_eqdsk.OMFITgeqdsk('/home/sciortino/Aurora/examples/example.gfile')
    inputgacode = omfit_gapy.OMFITgacode('/home/sciortino/Aurora/examples/example.input.gacode')
    
    # save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid

    rhop_kp = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
    ne = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
    Te = inputgacode['Te']*1e3  # keV --> eV

    # get charge state distributions from ionization equilibrium
    atom_data = aurora.atomic.get_atom_data(imp,['scd','acd'])
    logTe, fz, rates = aurora.get_frac_abundances(atom_data, ne, Te, rho=rhop_kp)

    if cs_den is None:
        # use ionization equilibrium fractional abundances as densities
        cs_den = fz
        rhop = rhop_kp
    else:
        # use provided charge state densities, given on rhop grid
        if rhop is None:
            raise ValueError('Which rhop grid were cs_dens arrays given on??')
        ne = interp1d(rhop_kp, ne, bounds_error=False, fill_value='extrapolate')(rhop)
        Te = interp1d(rhop_kp, Te, bounds_error=False, fill_value='extrapolate')(rhop)

        # normalize cs_den to match He-like density on axis
        cs_den /= cs_den[0,-3]/fz[0,-1]
        
    imp_Z = cs_den.shape[1] -1 

    # limit to core/pedestal
    ridx = np.argmin(np.abs(rhop - 0.99))
    ne = ne[:ridx]
    Te = Te[:ridx]
    rhop = rhop[:ridx]
    cs_den = cs_den[:ridx]

    # get w,z,x,y rate components
    out = compute_Helike_rates(imp_Z, ne, Te/1e3)   # Te input must be keV
    w_comps, z_comps, x_comps, y_comps = out

    # wavelengths for each line:
    w_lam = 3.1773e-10
    z_lam = 3.2111e-10 
    x_lam = 3.1892e-10
    y_lam = 3.1928e-10

    # conversion to frequency
    f_E_lam = lambda lam: h*c/(lam)

    n_H = cs_den[:,-2]
    n_He = cs_den[:,-3]
    n_Li = cs_den[:,-4]
    
    # compute line rates in phot/s/cm^3
    nw =  n_Li*w_comps[0] + n_He*w_comps[1] + n_H*w_comps[2]
    nz= n_Li*z_comps[0] + n_He*z_comps[1] + n_H*z_comps[2]
    nx = n_He*x_comps[0] + n_H*x_comps[1] + nz*x_comps[2]
    ny = n_He*y_comps[0] + n_H*y_comps[1] + nz*y_comps[2]

    # convert rates to J/s/cm^3
    w = f_E_lam(w_lam) * nw
    z = f_E_lam(z_lam) * nz
    x = f_E_lam(x_lam) * nx
    y = f_E_lam(y_lam) * ny

    # compute atomic plasma diagnostics only in the plasma core/pedestal
    R_ne = z / (x+y)
    G_Te = (z + x + y)/w

    # plot each line emissivity

    if axs is not None:
        ax0,ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]
        ls='--'
    else:

        if plot_individual_contributions:
            # make use of extra side space for labels
            fig = plt.figure(figsize=(11,7))    
            ax0 = plt.subplot2grid((5,5),(0,0), rowspan = 5, colspan=4)
            ax1 = plt.subplot2grid((5,5),(0,4), rowspan = 5, colspan=1, sharex=ax0)
        else:
            fig, ax0 = plt.subplots()
            ax1=None #dummy

        fig,ax2 = plt.subplots()
        fig,ax3 = plt.subplots()
        ls='-'

    lw=None

    # emissivity profiles
    ax0.plot(rhop, w, c='b', label='w', ls=ls, lw=lw)
    ax0.plot(rhop, z, c='r', label='z', ls=ls, lw=lw)
    ax0.plot(rhop, x, c='g', label='x', ls=ls, lw=lw)
    ax0.plot(rhop, y, c='m', label='y', ls=ls, lw=lw)
    ax0.set_xlabel(r'$\rho_p$')
    ax0.set_ylabel('IE emissivity [A.U.]')

    if plot_individual_contributions:
        ax0.plot(rhop, f_E_lam(w_lam) *n_Li*w_comps[0], ls='--', c='b', label='ionization')
        ax0.plot(rhop, f_E_lam(w_lam) *n_He*w_comps[1], ls=':', c='b', label='excitation', lw=lw)
        ax0.plot(rhop, f_E_lam(w_lam) *n_H*w_comps[2], ls='-.', c='b', label='recombination')
        
        ax0.plot(rhop, f_E_lam(z_lam) *n_Li*z_comps[0], ls='--', c='r', label='ionization')
        ax0.plot(rhop, f_E_lam(z_lam) *n_He*z_comps[1], ls=':', c='r', label='excitation', lw=lw)
        ax0.plot(rhop, f_E_lam(z_lam) *n_H*z_comps[2], ls='-.', c='r', label='recombination')
        
        ax0.plot(rhop, f_E_lam(x_lam) *n_He*x_comps[0], ls=':', c='g', label='excitation', lw=lw)
        ax0.plot(rhop, f_E_lam(x_lam) *n_H*x_comps[1], ls='-.', c='g', label='recombination')
        ax0.plot(rhop, f_E_lam(x_lam) *nz*x_comps[2], marker='*', c='g', label='z-prop')
        
        ax0.plot(rhop, f_E_lam(y_lam) *n_He*y_comps[0], ls=':', c='m', label='excitation', lw=lw)
        ax0.plot(rhop, f_E_lam(y_lam) *n_H*y_comps[1], ls='-.', c='m', label='recombination')
        ax0.plot(rhop, f_E_lam(y_lam) *nz * y_comps[2], marker='*', c='m', label='z-prop')
        

    if axs is None: # if axes were given, no need to re-plot labels
        if plot_individual_contributions:    
            
            # basic labels for each line
            ax1.plot([],[], c='b', label='w')
            ax1.plot([],[], c='r', label='z')
            ax1.plot([],[], c='g', label='x')
            ax1.plot([],[], c='m', label='y')

            # show labels for each contribution
            ax1.plot([],[], c='w', label=' ') #empty to separate colors and line styles
            ax1.plot([],[], ls='--', c='k', label='ionization', lw=lw)
            ax1.plot([],[], ls=':', c='k', label='excitation', lw=lw)
            ax1.plot([],[], ls='-.', c='k', label='recombination', lw=lw)
            ax1.plot([],[], marker='*', c='k', label='z cascade', lw=lw)
            
            leg = ax1.legend(loc='center left').set_draggable(True)
            ax1.axis('off')
        else:
            leg = ax0.legend().set_draggable(True)

    plt.tight_layout()

    # plot line ratios to w
    ax2.plot(rhop, w/w, label='w', ls=ls, c='b')
    ax2.plot(rhop, z/w, label='z', ls=ls, c='r')
    ax2.plot(rhop, x/w, label='x', ls=ls, c='g')
    ax2.plot(rhop, y/w, label='y', ls=ls, c='m')
    ax2.set_yscale('log')
    if axs is None: # if axes were given, no need to re-plot labels
        ax2.legend().set_draggable(True)
        ax2.set_xlabel(r'$\rho_p$')
        ax2.set_ylabel('Line ratios to w')
        
        # set good tick frequency on log-scale for comparison (a bit ad-hoc..)
        ax2.set_yticks([0.1,0.3, 1.0, 3.0, 6.0])
        ax2.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.tight_layout()

    # plot atomic plasma diagnostics
    ax3.plot(rhop, R_ne, label=r'R($n_e$)', ls=ls, c='b')
    ax3.plot(rhop, G_Te, label=r'G($T_e$)', ls=ls, c='r')
    if axs is None:
        leg = ax3.legend().set_draggable(True)
        ax3.set_xlabel(r'$\rho_p$')

        # set good tick frequency on log-scale for comparison (a bit ad-hoc..)
        ax3.set_yticks([0.1,0.3, 1.0, 3.0, 6.0])
        ax3.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.tight_layout()

    return [ax0,ax1], ax2, ax3




def plot_wz_case(ion, Z, dens_grid, temp_grid, fig_stuff=None):
    ''' Plot components of w and z rates '''
    w_comps, z_comps, x_comps, y_comps = compute_Helike_rates(Z, dens_grid, temp_grid)
    
    if fig_stuff is None:
        fig,ax1 = plt.subplots(figsize=(10,6))
    else:
        fig,ax1 = fig_stuff

    ax1.set_ylabel('w line', color='r')
    ax1.loglog(temp_grid, w_comps[0], 'r-')
    ax1.loglog(temp_grid, w_comps[1], 'r--')
    ax1.loglog(temp_grid, w_comps[2], 'r:')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.set_ylabel('z line', color='b')
    ax2.loglog(temp_grid, z_comps[0], 'b-')
    ax2.loglog(temp_grid, z_comps[1], 'b--')
    ax2.loglog(temp_grid, z_comps[2], 'b:')
    ax2.tick_params(axis='y', labelcolor='b')

    ax1.set_xlabel('Temperature [keV]')
    fig.suptitle(f'He-like {ion} PEC components')

    # for legend
    ax1.loglog([],[], 'k-', label='ionization')
    ax1.loglog([],[], 'k--', label='excitation')
    ax1.loglog([],[], 'k:', label='recombination')

    ax1.legend(loc='best').set_draggable(True)




def plot_line_case(ion, Z, ne_grid, Te_grid, color, fig_stuff=None, spec_line='z'):
    ''' Plot components of w rates '''
    w_comps, z_comps, x_comps, y_comps = compute_Helike_rates(Z, ne_grid, Te_grid)
    
    if fig_stuff is None:
        fig = plt.figure(figsize=(10,6))
        fig.set_size_inches(10,7, forward=True)
        ax1 = plt.subplot2grid((1,10),(0,0),rowspan = 1, colspan = 8)
        ax2 = plt.subplot2grid((1,10),(0,8),rowspan = 1, colspan = 2)
        ax2.axis('off')

        # for legend
        ax2.loglog([],[], 'k-', label='ionization')
        ax2.loglog([],[], 'k--', label='excitation')
        ax2.loglog([],[], 'k:', label='recombination')
        ax1.set_xlabel('Temperature [keV]')
        fig.suptitle(f'He-like {ion} PEC components')

        ax2.legend(loc='best').set_draggable(True)
        ax1.set_ylabel(f'{spec_line} line')

    else:
        fig,ax1,ax2 = fig_stuff
        
    line_comps = w_comps if spec_line=='w' else z_comps
    ax1.loglog(temp_grid, line_comps[0], ls='-', color=color)
    ax1.loglog(temp_grid, line_comps[1], ls='--', color=color)
    ax1.loglog(temp_grid, line_comps[2], ls=':', lw=2, color=color)
    ax2.loglog([],[], color=color, label=f'$n_e={ne_grid[0]:.1e}$ $cm^{{{-3}}}$')

    ax2.legend(loc='best', fontsize=14).set_draggable(True)
    return fig,ax1,ax2





if __name__=='__main__':

    import periodictable
    from matplotlib.pyplot import cm

    # ions for which we have info in atdata.dat:
    #Zs = [2, 5, 6, 7, 8, 9, 10, 13, 17, 18, 20, 21, 22, 26, 28, 29, 36, 42]

    element_symbols = np.asarray([el.symbol for el in periodictable.elements])
    
    # input Te must be in keV:
    temp_grid = np.geomspace(1e-1, 10, 1000) # keV
    dens_grid = np.ones_like(temp_grid) *1e13  # cm^-3
    
    for ion in ['O','Mg','Ar','Fe']: #element_symbols:
        Z = np.where(element_symbols==ion)[0][0]

        try:
            plot_wz_case(ion, Z, dens_grid, temp_grid)
        except:
            pass

    # just for single ion
    cols = cm.rainbow(np.linspace(0,1,3))

    # w line:
    fig_stuff = plot_line_case('Ca', 20, np.ones_like(temp_grid) *1e12, temp_grid, cols[0], spec_line='w')
    fig_stuff = plot_line_case('Ca', 20, np.ones_like(temp_grid) *1e13, temp_grid, cols[1], fig_stuff=fig_stuff, spec_line='w')
    fig_stuff = plot_line_case('Ca', 20, np.ones_like(temp_grid) *1e14, temp_grid, cols[2], fig_stuff=fig_stuff, spec_line='w')

    # z line:
    fig_stuff = plot_line_case('Ca', 20, np.ones_like(temp_grid) *1e12, temp_grid, cols[0], spec_line='z')
    fig_stuff = plot_line_case('Ca', 20, np.ones_like(temp_grid) *1e13, temp_grid, cols[1], fig_stuff=fig_stuff, spec_line='z')
    fig_stuff = plot_line_case('Ca', 20, np.ones_like(temp_grid) *1e14, temp_grid, cols[2], fig_stuff=fig_stuff, spec_line='z')



    ###
    #axs = Helike_emiss_metrics(imp='Ca')
    #axs = Helike_emiss_metrics(imp='O')
