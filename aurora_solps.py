import pickle as pkl
import matplotlib.pyplot as plt
import MDSplus
import numpy as np, copy, scipy
plt.ion()
from scipy.interpolate import griddata,interp2d
from matplotlib import ticker, cm
import aurora
import omfit_eqdsk
import matplotlib.tri as tri
from heapq import nsmallest

from IPython import embed


class solps_case:
    def __init__(self,path, case_num, geqdsk):

        self.path = path
        self.case_num = case_num
        
        if isinstance(geqdsk, str):
            # user passed a path to a gfile on disk
            self.geqdsk = omfit_eqdsk.OMFITgeqdsk(geqdsk)
        else:
            self.geqdsk = geqdsk
    
        # separate core, SOL and PFR mesh points
        self.radloc = np.loadtxt(path+f'/Output/RadLoc{case_num}')
        self.unit_p = int(np.max(self.radloc[:,0])//4)
        self.unit_r = int(np.max(self.radloc[:,1])//2)

        # Obtain indices for chosen radial regions
        R_idxs = np.array([],dtype=int)
        R_idxs = np.concatenate((R_idxs, np.arange(self.unit_r+1)))
        self.R_idxs = np.concatenate((R_idxs, np.arange(self.unit_r+1,2*self.unit_r+2)))

        # obtain indices for chosen poloidal regions
        P_idxs = np.array([],dtype=int)
        P_idxs = np.concatenate((P_idxs, np.arange(1,self.unit_p+1)))  # Inner PFR
        P_idxs = np.concatenate((P_idxs, np.arange(self.unit_p+1,3*self.unit_p+1)))  # core/open SOL
        self.P_idxs = np.concatenate((P_idxs, np.arange(3*self.unit_p+1,4*self.unit_p+1)))  # outer PFR


    def load_solps_data(self, field_names=None, P_idxs=None, R_idxs=None,
                        Rmin=None, Rmax=None, Pmin=None, Pmax=None):
        '''Load SOLPS output from files for each of the needed quantities.

        Keyword Args:
            field_names : list or array
                List of fields to fetch from SOLPS output. If left to None, by default uses
                ['Ne','Te','NeuDen','NeuTemp','MolDen','MolTemp','Ti']
            P_idxs : list or array
                Poloidal indices to load.
            R_idxs : list or array
                Radial indices to load.
            Rmin : int or None.
                Minimum major radius index to load, if R_idxs is not given
            Rmax : int or None
                Maximum major radius index to load, if R_idxs is not given
            Pmin : int or None
                Minimum poloidal index to load, if P_idxs is not given
            Pmax : int or None
                Maximum poloidal index to load, if P_idxs is not given
        
        Returns:
            quants : dict
                Dictionary containing 'R','Z' coordinates for 2D maps of each field requested by user.
        '''
        
        RDIM = int(np.max(self.radloc[:,1]))+2
        PDIM = int(np.max(self.radloc[:,0]))+2

        if P_idxs is None:
            if Pmax is None: Pmax = PDIM
            if Pmin is None: Pmin = 0
            P_idxs = np.arange(Pmin,Pmax)
        else:
            pass # user provided list of poloidal grid indices

        if R_idxs is None:
            if Rmax is None: Rmax = RDIM
            if Rmin is None: Rmin = 0
            R_idxs = np.arange(Rmin,Rmax)
        else:
            pass # user provided list of radial grid indices

        quants = {}

        self.R = quants['R'] = np.atleast_2d(
            np.loadtxt(self.path+f'Output/RadLoc{self.case_num}',
                       usecols=(3)).reshape((RDIM,PDIM))[R_idxs,:])[:,P_idxs]
        self.Z = quants['Z'] = np.atleast_2d(
            np.loadtxt(self.path+f'Output/VertLoc{self.case_num}',
                       usecols=(3)).reshape((RDIM,PDIM))[R_idxs,:])[:,P_idxs]

        if field_names is None:
            field_names = ['Ne','Te','NeuDen','NeuTemp','MolDen','MolTemp','Ti']

        for field_name in field_names:
            tmp =  np.loadtxt(self.path+f'Output/{field_name}{self.case_num}',
                                       usecols=(3)).reshape((RDIM,PDIM))
            quants[field_name] = np.atleast_2d(tmp[R_idxs,:])[:,P_idxs]

        return quants

    
    def process_solps_data(self, fields=None, P_idxs=None, R_idxs=None,
                           edge_max=0.05, plot=False):
        '''Load and process SOLPS to permit clear plotting. 
        
        Keyword Args:
            fields : dict
                Dictionary containing SOLPS outputs to process. Keys indicate the quantity, value its label
                (only used for plotting). If left to None, defaults fields of 'Ne','Te','NeuDen','NeuTemp'
                are used.
            P_idxs : list or array
                Poloidal indices to load.
            R_idxs : list or array
                Radial indices to load.
            edge_max : float
                Maximum edge length in triangulation. Setting this parameter allows improvements in 
                visualization of the SOLPS grid regions.
            plot : bool
                If True, plot results for all loaded 2D quantities. 

        Returns:
            quants : dict
                Dictionary containing 'R','Z' coordinates for 2D maps of each field requested by user.
                Quantities are processed and masked to facilitate plotting.
        '''
        if fields is None:
            fields = {'Ne':r'log($n_e$) [$m^{-3}$]' , 'Te':r'log($T_e$) [eV]',
                      'NeuDen':'log($n_n$) [$m^{-3}$]' , 'NeuTemp':'log($T_n$) [eV]',
                      #'MolDen':'log($n_m$) [$m^{-3}$]' , 'MolTemp':'log($T_m$) [eV]',
                      #'Ti':r'log($T_i$) [eV]',
            }
        if P_idxs is None:
            P_idxs = self.P_idxs
        if R_idxs is None:
            R_idxs = self.R_idxs

        self.quants = self.load_solps_data(P_idxs=P_idxs, R_idxs=R_idxs)

        # apply masking based on maximum side-length
        self.triang = apply_mask(self.quants, self.geqdsk, dist=0.05)   # 0.06 creates some long segments at top

        # set zero densities equal to min to avoid log issues
        self.quants['NeuDen'][self.quants['NeuDen']==0.0] = nsmallest(2,np.unique(self.quants['NeuDen'].flatten()))[1]

        if plot:

            fig,axs = plt.subplots(2,2, figsize=(9,12),sharex=True) 
            ax = axs.flatten()

            cbars = [];
            for ii,field in enumerate(fields):
                label = fields[field]

                ax[ii].plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
                cntr = ax[ii].tricontourf(self.triang, np.log10(self.quants[field]).flatten(),
                                           cmap=cm.magma,levels=300)

                # create draggable colorbar
                cbars.append( plt.colorbar(cntr, format='%.3g', ax=ax[ii]))
                cbars[-1].ax.set_title(label)
                cbars[-1] = aurora.DraggableColorbar(cbars[-1],cntr)
                cbars[-1].connect()

                ax[ii].axis('equal')
                ax[ii].set_xlabel('R [m]')
                ax[ii].set_ylabel('Z [m]')



    def plot_2d_vals(self, vals, label='', mask_opt={'up':False,'down':True}):
        '''Method to plot 2D fields over the R and Z grids, internally stored. 

        The "mask" dictionary allows one to mask the upper or lower region, if useful. 

        '''
        #mask upper or lower divertor region:
        # R = copy.deepcopy(self.R)
        # Z = copy.deepcopy(self.Z)
        # vals_m = copy.deepcopy(vals);
        # if mask_opt['up']:
        #     mask = self.Z>np.max(self.geqdsk['ZBBBS'])
        #     R[mask]= np.nan
        #     Z[mask] = np.nan
        #     vals_m[mask] = np.nan
        # if mask_opt['down']:
        #     mask = self.Z<np.min(self.geqdsk['ZBBBS'])
        #     R[mask]= np.nan
        #     Z[mask] = np.nan
        #     vals_m[mask] = np.nan
            

        # plot entire field with nice triangulation:
        fig,ax0 = plt.subplots(figsize=(8,11))
        ax0.axis('equal')
        ax0.set_xlabel('R [m]'); ax0.set_ylabel('Z [m]')
        ax0.plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
        cntr = ax0.tricontourf(self.triang, vals.flatten(), cmap=cm.magma,levels=300)
        cbar = plt.colorbar(cntr, format='%.3g', ax=ax0)
        cbar = aurora.DraggableColorbar(cbar,cntr)
        ax0.set_title(label)
        cbar.connect()

        # # plot only masked region (triangulation not set up, so grid edges will be visible)
        # fig,axx = plt.subplots(num=label, figsize=(8,11))
        # axx.axis('equal')
        # axx.set_xlabel('R [m]')
        # axx.set_ylabel('Z [m]')
        # axx.plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
        # cntr = axx.contourf(self.R, self.Z, vals_m, cmap=cm.magma,levels=300)
        # cbar = plt.colorbar(cntr, format='%.3g', ax=axx)
        # cbar = aurora.DraggableColorbar(cbar,cntr)
        # axx.set_title(label)
        # cbar.connect()   

    
    def get_n0_profiles(self):
        '''Extract atomic neutral density profiles from the SOLPS run. 
        This method returns neutral densities both at the low field side (LFS) and as a 
        flux surface average (FSA). 
        '''
        rhop_2D = aurora.get_rhop_RZ(self.quants['R'],self.quants['Z'], self.geqdsk)
        
        # evaluate FSA neutral densities inside the LCFS
        def avg_function(r,z):
            if any(aurora.get_rhop_RZ(r,z, self.geqdsk)<np.min(rhop_2D)):
                return np.nan
            else:
                return griddata((self.quants['R'].flatten(),self.quants['Z'].flatten()), self.quants['NeuDen'].flatten(),
                                (r,z), method='cubic')

        neut_fsa = self.geqdsk['fluxSurfaces'].surfAvg(function=avg_function)
        rhop_fsa = np.sqrt(self.geqdsk['fluxSurfaces']['geo']['psin'])

        # get R axes on the midplane on the LFS and HFS
        mask = (self.quants['Z'].flatten()>-0.02)&(self.quants['Z'].flatten()<0.02)
        R_midplane = self.quants['R'].flatten()[mask]
        
        R_midplane_lfs = R_midplane[R_midplane>self.geqdsk['RMAXIS']]
        R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),100)
        
        R_midplane_hfs = R_midplane[R_midplane<self.geqdsk['RMAXIS']]
        R_HFS = np.linspace(np.min(R_midplane_hfs), np.max(R_midplane_hfs),100)
        
        # get neutral densities at midradius (LFS and HFS)
        neut_LFS = griddata((self.quants['R'].flatten(),self.quants['Z'].flatten()), self.quants['NeuDen'].flatten(),
                                 (R_LFS,np.zeros_like(R_LFS)), method='cubic')
        rhop_LFS = aurora.get_rhop_RZ(R_LFS,np.zeros_like(R_LFS), self.geqdsk)
        
        neut_HFS = griddata((self.quants['R'].flatten(),self.quants['Z'].flatten()), self.quants['NeuDen'].flatten(),
                                 (R_HFS,np.zeros_like(R_HFS)), method='cubic')
        rhop_HFS = aurora.get_rhop_RZ(R_HFS,np.zeros_like(R_HFS), self.geqdsk)
        
        # compare FSA neutral densities with midplane ones
        fig,ax = plt.subplots()
        ax.plot(rhop_fsa, neut_fsa, label='FSA')
        ax.plot(rhop_LFS, neut_LFS, label='outboard midplane')
        ax.plot(rhop_HFS, neut_HFS, label='inboard midplane')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(r'$n_0$ [$m^{-3}$]')
        ax.legend(loc='best').set_draggable(True)
        plt.tight_layout()

        return rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS
        
def apply_mask(quants, geqdsk, dist=0.4, mask_up=False, mask_down=True):
    '''Function to mask triangulation with edges longer than "dist". 

    If mask_up=True, the mask is applied to the upper vertical (y) half of the mesh. 
    If mask_down=True, the mask is applied in the lower vertical (y) part of the mesh.
    '''
    x = quants['R'].flatten()
    y = quants['Z'].flatten()
    triang = tri.Triangulation(x, y)

    # Mask triangles with sidelength bigger some dist
    triangles = triang.triangles

    # Find triangles with sides longer than dist
    xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
    ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
    cond_maxd = maxi > dist

    # second condition: mask upper and/or lower part of the grid
    if mask_up:
        cond_up = np.mean(y[triangles],axis=1)>0.0
    else:
        cond_up = np.mean(y[triangles],axis=1)<1e10  # all True
    if mask_down:
        cond_down = np.mean(y[triangles],axis=1)<=0.0
    else:
        cond_down = np.mean(y[triangles],axis=1)<1e10  # all True

    cond_y = cond_up & cond_down

    # apply mask to points within the LCFS (no triangulation issues here)
    rhop_triangs = aurora.get_rhop_RZ(np.mean(x[triangles],axis=1),
                             np.mean(y[triangles],axis=1),
                             geqdsk) 
    center_mask = rhop_triangs < np.min(aurora.get_rhop_RZ(x,y,geqdsk))

    # apply masking
    triang.set_mask((cond_maxd & cond_y) | center_mask)

    return triang


def overplot_machine(shot, ax):
    # ----------------------------#
    # Overplot machine structures #
    # ----------------------------#
    MDSconn = MDSplus.Connection('alcdata.psfc.mit.edu')

    #pull cross-section from tree
    try:
        ccT = MDSconn.openTree('analysis',shot)
        path = '\\analysis::top.limiters.tiles:'
        xvctr = MDSconn.get(path+'XTILE').data()
        yvctr = MDSconn.get(path+'YTILE').data()
        nvctr = MDSconn.get(path+'NSEG').data()
        lvctr = MDSconn.get(path+'PTS_PER_SEG').data()
    except:
        raise ValueError('data load failed.')

    x = []
    y = []
    for i in range(nvctr):
        length = lvctr[i]
        xseg = xvctr[i,0:length]
        yseg = yvctr[i,0:length]
        x.extend(xseg)
        y.extend(yseg)
        if i != nvctr-1:
            x.append(None)
            y.append(None)

    x = np.array(x); y = np.array(y)

    for ii in np.arange(len(ax)):
        ax[ii].plot(x,y)


        








    
if __name__=='__main__':


    #from aurora_solps import *
    shot = 1120917011
    case_num = 68 #4
    path = f'/home/sciortino/SOLPS/RR_{shot}_attempts/Attempt{case_num}/'

    # path to geqdsk file
    gfilepath = '/home/sciortino/EFIT/g1120917011.00999_981'  # hard-coded

    # load SOLPS results
    solps_out = solps_case(path, case_num, gfilepath)

    # fetch and process SOLPS output 
    solps_out.process_solps_data(plot=False, edge_max=0.05)
    quants = solps_out.quants
    
    solps_out.plot_2d_vals(quants['NeuDen'], label=r'$n_n$', mask_opt={'up':False,'down':True})
    
    # Obtain impurity charge state predictions from ioniz equilibrium
    imp = 'C'
    ne_cm3 = quants['Ne'] *1e-6  #cm^-3
    n0_cm3 = quants['NeuDen'] *1e-6 #cm^-3
    Te_eV = quants['Te']  # eV
    filetypes=['acd','scd','ccd']
    filenames = []
    atom_data = aurora.get_atom_data(imp,filetypes,filenames)
    
    logTe, fz, rate_coeffs = aurora.get_frac_abundances(
        atom_data,ne_cm3,Te_eV, n0_by_ne=n0_cm3/ne_cm3, plot=False)  # added n0

    frac=0.01  # 1%
    nz_cm3 = frac * ne_cm3[:,:,None] * fz  # (time,nZ,space) --> (R,nZ,Z)
    nz_cm3 = nz_cm3.transpose(0,2,1)

    # compute radiation density from each grid point
    out = aurora.compute_rad(imp,nz_cm3, ne_cm3, Te_eV, n0=n0_cm3, Ti=Te_eV,
                             prad_flag=True,thermal_cx_rad_flag=True)

    # plot total radiated power
    solps_out.plot_2d_vals(out['tot']*1e3, label=r'$P_{rad}$ [$kW/m^3$]', mask_opt={'up':False,'down':False})
    
    # plot total line radiated power
    solps_out.plot_2d_vals(out['line_rad'].sum(1)*1e3, label=r'$P_{line,rad}$ [$kW/m^3$]',mask_opt={'up':False,'down':True})

    # overplot machine tiles
    overplot_machine(shot, [plt.gca()])
     
    # compare SOLPS results at midplane with FSA
    rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS = solps_out.get_n0_profiles()
