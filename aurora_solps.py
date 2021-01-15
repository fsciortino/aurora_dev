'''Aurora-SOLPS coupling methods.

sciortino, 2021
'''
import pickle as pkl
import matplotlib.pyplot as plt
import MDSplus, os
import numpy as np, copy, scipy
plt.ion()
from scipy.interpolate import griddata,interp2d
from matplotlib import ticker, cm
import aurora
import omfit_eqdsk
import matplotlib.tri as tri
from heapq import nsmallest
import omfit_solps

from IPython import embed

class solps_case:
    def __init__(self, path, geqdsk, solps_run='P29MW_n1.65e20_Rout0.9_Y2pc',
                 case_num=0, form='extracted'):
        '''Read SOLPS output and prepare for Aurora impurity-neutral analysis. 

        Args:
            path : str
                Path to output files. If these are "extracted" files from SOLPS (form='extracted'),
                then this is the path to the disk location where each of the required files
                can be found. These are
                [RadLoc, VertLoc, Ne, Te, NeuDen, NeuTemp, MolDen, MolTemp, Ti]{case_num}
                Otherwise, if form='full', this path indicates where to find the directory named "baserun"
                and the 'solps_run' one.
            geqdsk : str or `omfit_geqdsk` class instance
                Path to the geqdsk to load from disk, or instance of the `omfit_geqdsk` class that
                contains the processed gEQDSK file already. 

        Keyword Args:
            solps_run : str
                If form='full', this string specifies the directory (relative to the given path)
                where case-specific files for a SOLPS run can be found (e.g. 'b2fstate').
            case_num : int
                Index/integer identifying the SOLPS case of interest. 
            form : str
                Form of SOLPS output to be loaded, one of {'full','extracted'}. 
                The 'full' output consists of 'b2fstate', 'b2fgmtry', 'fort.44', etc.
                The 'extracted' output consists of individual files containing each quantity in separate
                files. 
        '''

        self.path = path
        self.solps_run = solps_run
        self.case_num = case_num
        self.form = form
        
        if isinstance(geqdsk, str):
            # user passed a path to a gfile on disk
            self.geqdsk = omfit_eqdsk.OMFITgeqdsk(geqdsk)
        else:
            self.geqdsk = geqdsk

        if form=='extracted':
            self.radloc = np.loadtxt(path+f'/RadLoc{case_num}')
            self.unit_p = int(np.max(self.radloc[:,0])//4)
            self.unit_r = int(np.max(self.radloc[:,1])//2)

            self.nx = int(np.max(self.radloc[:,1]))+2
            self.ny = int(np.max(self.radloc[:,0]))+2
            
        elif form=='full':
            self.b2fstate = omfit_solps.OMFITsolps(path+os.sep+solps_run+os.sep+'b2fstate')
            self.geom = omfit_solps.OMFITsolps(path+os.sep+'baserun'+os.sep+'b2fgmtry')

            self.nx,self.ny = self.geom['nx,ny']
            
            # (0,:,:): lower left corner, (1,:,:): lower right corner
            # (2,:,:): upper left corner, (3,:,:): upper right corner.
            
            self.crx = self.geom['crx'].reshape(4,self.ny+2,self.nx+2)  # horizontal 
            self.cry = self.geom['cry'].reshape(4,self.ny+2,self.nx+2)  # vertical
            
            # uncertain units for splitting of radial and poloidal regions...
            self.unit_p = self.cry.shape[2]//4
            self.unit_r = self.crx.shape[1]//2 #-1
            
        # Obtain indices for chosen radial regions
        R_idxs = np.array([],dtype=int)
        R_idxs = np.concatenate((R_idxs, np.arange(self.unit_r+1)))
        self.R_idxs = np.concatenate((R_idxs, np.arange(self.unit_r+1,2*self.unit_r+2)))

        # obtain indices for chosen poloidal regions
        P_idxs = np.array([],dtype=int)
        P_idxs = np.concatenate((P_idxs, np.arange(1,self.unit_p+1)))  # Inner PFR
        P_idxs = np.concatenate((P_idxs, np.arange(self.unit_p+1,3*self.unit_p+1)))  # core/open SOL
        self.P_idxs = np.concatenate((P_idxs, np.arange(3*self.unit_p+1,4*self.unit_p+1)))  # outer PFR


    def load_data(self, field_names=None, P_idxs=None, R_idxs=None,
                  Rmin=None, Rmax=None, Pmin=None, Pmax=None):
        '''Load SOLPS output for each of the needed quantities ('extracted' form)

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
        if P_idxs is None:
            if Pmax is None: Pmax = self.ny
            if Pmin is None: Pmin = 0
            P_idxs = np.arange(Pmin,Pmax)
        else:
            pass # user provided list of poloidal grid indices

        if R_idxs is None:
            if Rmax is None: Rmax = self.nx
            if Rmin is None: Rmin = 0
            R_idxs = np.arange(Rmin,Rmax)
        else:
            pass # user provided list of radial grid indices

        if field_names is None:
            field_names = ['Ne','Te','NeuDen','NeuTemp','MolDen','MolTemp','Ti']
            
        self.quants = quants = {}

        if self.form=='extracted':
            self.R = np.atleast_2d(
                np.loadtxt(self.path+f'/RadLoc{self.case_num}',
                           usecols=(3)).reshape((self.nx,self.ny))[R_idxs,:])[:,P_idxs]
            self.Z = np.atleast_2d(
                np.loadtxt(self.path+f'/VertLoc{self.case_num}',
                           usecols=(3)).reshape((self.nx,self.ny))[R_idxs,:])[:,P_idxs]
            
            for field_name in field_names:
                tmp =  np.loadtxt(self.path+f'/{field_name}{self.case_num}',
                                  usecols=(3)).reshape((self.nx,self.ny))
                quants[field_name] = np.atleast_2d(tmp[R_idxs,:])[:,P_idxs]

            self.triang = tri.Triangulation(self.R.flatten(), self.Z.flatten())
            self.triang_e = self.triang # triangulation for EIRENE is mapped to b2 grid
    
        elif self.form=='full':

            self.R = np.mean(self.crx,axis=0)[1:-1,:][:,1:-1]# [R_idxs[1:-1],:][:,P_idxs]
            self.Z = np.mean(self.cry,axis=0)[1:-1,:][:,1:-1]#[R_idxs[1:-1],:][:,P_idxs]
            
            if 'Ne' in field_names:
                quants['Ne'] = self.b2fstate['ne']
            if 'Te' in field_names:
                quants['Te'] = self.b2fstate['te']
            if 'Ti' in field_names:
                quants['Ti'] = self.b2fstate['ti']

            self.eirene_out = self.load_eirene_output(files=['fort.44'])  #'fort.46'

            ########
            ## Read on EIRENE mesh
            # molecular density and temperature
            # quants['MolDen'] = self.eirene_out['fort.46']['pdenm']
            # quants['MolTemp']= self.eirene_out['fort.46']['edenm']

            # # group atomic neutral density from all H isotopes
            # spmask = self.b2fstate['zn']==1
            # quants['NeuDen'] = np.sum(
            #     self.eirene_out['fort.46']['pdena'].reshape(-1,sum(spmask)),
            #     axis=1)

            # quants['NeuTemp'] = np.sum(
            #     self.eirene_out['fort.46']['edena'].reshape(-1,sum(spmask)),
            #     axis=1)
            ####

            tmp = self.eirene_out['fort.44']['dmb2'].reshape((self.nx,self.ny))
            quants['MolDen'] = np.atleast_2d(tmp).T#[R_idxs[1:-1],:])[:,P_idxs]
            tmp = self.eirene_out['fort.44']['tmb2'].reshape((self.nx,self.ny))
            quants['MolTemp'] = np.atleast_2d(tmp).T#[R_idxs[1:-1],:])[:,P_idxs]

            # group atomic neutral density from all H isotopes
            spmask = self.b2fstate['zn']==1
            tmp = np.sum(
                self.eirene_out['fort.44']['dab2'].reshape(-1,sum(spmask)),
                axis=1).reshape((self.nx,self.ny))
            quants['NeuDen'] = np.atleast_2d(tmp).T#[R_idxs,:])[:,P_idxs]
            tmp = np.sum(
                self.eirene_out['fort.44']['tab2'].reshape(-1,sum(spmask)),
                axis=1).reshape((self.nx,self.ny))
            quants['NeuTemp'] = np.atleast_2d(tmp).T#[R_idxs,:])[:,P_idxs]
            
            self.triang = tri.Triangulation(self.R.flatten(), self.Z.flatten())
            self.triang_e = self.eirene_out['triang']
            
            
            


    def load_eirene_output(self, files = ['fort.44','fort.46']):
        '''Load result from one of the fort.* files with EIRENE output, 
        as indicated by the "files" list argument.

        Keyword Args:
            files : list or array-like
                EIRENE output files to read. Default is to load all
                files for which this method has been tested. 

        Returns:
            eirene_out : dict
                Dictionary for each loaded file containing a subdictionary
                with keys for each loaded field from each file. 

        '''
        eirene_out = {}
        
        for filename in files: 
            eirene_out[filename] = {}
            # load each of these files into dictionary structures
            with open(self.path +os.sep+self.solps_run+os.sep+filename, 'r') as f:
                contents = f.readlines()
            ii=6

            while ii<len(contents[ii:]):
                if  contents[ii].startswith('*eirene'):
                    key = contents[ii].split()[3]
                    eirene_out[filename][key] = []
                else:
                    [eirene_out[filename][key].append(float(val)) for val in contents[ii].strip().split()]
                ii+=1

            for key in eirene_out[filename]:
                eirene_out[filename][key] = np.array(eirene_out[filename][key])

        # Now load fort.33 file with EIRENE nodes and cells
        Nodes=np.fromfile(self.path+os.sep+'baserun'+os.sep+'fort.33',sep=' ')
        NN=int(Nodes[0])
        XNodes=Nodes[1:NN+1]/100  # cm -->m
        YNodes=Nodes[NN+1:]/100
        
        # EIRENE triangulation
        Triangles = np.loadtxt(
            self.path+os.sep+'baserun'+os.sep+'fort.34',skiprows=1, usecols=(1,2,3))

        eirene_out['triang'] =tri.Triangulation(XNodes,YNodes,triangles=(Triangles-1))

        return eirene_out


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

        # load data arrays
        self.load_data(field_names = list(fields.keys()), P_idxs=P_idxs, R_idxs=R_idxs)

        # apply masking based on maximum side-length
        self.triang = apply_mask(self.triang, self.geqdsk, dist=0.05)   # 0.06 creates some long segments at top

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



    def plot_2d_vals(self, vals, label=''):
        '''Method to plot 2D fields over the R and Z grids, internally stored. 

        The "mask" dictionary allows one to mask the upper or lower region, if useful. 

        '''
        # plot entire field with nice triangulation:
        fig,ax0 = plt.subplots(figsize=(8,11))
        ax0.axis('equal')
        ax0.set_xlabel('R [m]'); ax0.set_ylabel('Z [m]')
        ax0.plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
        # triangulation from b2 R and Z grid
        cntr = ax0.tricontourf(self.triang, vals.flatten(), cmap=cm.magma,levels=300)
        cbar = plt.colorbar(cntr, format='%.3g', ax=ax0)
        cbar = aurora.DraggableColorbar(cbar,cntr)
        ax0.set_title(label)
        cbar.connect()

        ####
        fig,ax0 = plt.subplots(figsize=(8,11))
        ax0.axis('equal')
        ax0.set_xlabel('R [m]'); ax0.set_ylabel('Z [m]')
        ax0.plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
        ax0.contourf(self.R, self.Z, vals)
        cbar = plt.colorbar(cntr, format='%.3g', ax=ax0)
        cbar = aurora.DraggableColorbar(cbar,cntr)
        ax0.set_title(label)
        cbar.connect()
    
    def get_n0_profiles(self):
        '''Extract atomic neutral density profiles from the SOLPS run. 
        This method returns neutral densities both at the low field side (LFS) and as a 
        flux surface average (FSA). 
     
        '''
        rhop_2D = aurora.get_rhop_RZ(self.R,self.Z, self.geqdsk)
        
        # evaluate FSA neutral densities inside the LCFS
        def avg_function(r,z):
            if any(aurora.get_rhop_RZ(r,z, self.geqdsk)<np.min(rhop_2D)):
                return np.nan
            else:
                return griddata((self.R.flatten(),self.Z.flatten()), self.quants['NeuDen'].flatten(),
                                (r,z), method='nearest')

        neut_fsa = self.geqdsk['fluxSurfaces'].surfAvg(function=avg_function)
        rhop_fsa = np.sqrt(self.geqdsk['fluxSurfaces']['geo']['psin'])

        # get R axes on the midplane on the LFS and HFS
        dz = (np.max(self.Z) - np.min(self.Z))/((self.nx+self.ny)/10.) # rule-of-thumb to identify vertical resolution
        mask = (self.Z.flatten()>-dz)&(self.Z.flatten()<dz)
        R_midplane = self.R.flatten()[mask]
        
        R_midplane_lfs = R_midplane[R_midplane>self.geqdsk['RMAXIS']]
        R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),100)
        
        R_midplane_hfs = R_midplane[R_midplane<self.geqdsk['RMAXIS']]
        R_HFS = np.linspace(np.min(R_midplane_hfs), np.max(R_midplane_hfs),100)
        
        # get neutral densities at midradius (LFS and HFS)
        neut_LFS = griddata((self.R.flatten(),self.Z.flatten()), self.quants['NeuDen'].flatten(),
                                 (R_LFS,np.zeros_like(R_LFS)), method='cubic')
        rhop_LFS = aurora.get_rhop_RZ(R_LFS,np.zeros_like(R_LFS), self.geqdsk)
        
        neut_HFS = griddata((self.R.flatten(),self.Z.flatten()), self.quants['NeuDen'].flatten(),
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
        
def apply_mask(triang, geqdsk, dist=0.4, mask_up=False, mask_down=True):
    '''Function to mask triangulation with edges longer than "dist". 

    If mask_up=True, the mask is applied to the upper vertical (y) half of the mesh. 
    If mask_down=True, the mask is applied in the lower vertical (y) part of the mesh.
    '''
    # Mask triangles with sidelength bigger some dist
    triangles = triang.triangles
    x = triang.x; y = triang.y
    
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


    case='CMOD'
    if case=='CMOD':
        shot = 1120917011
        case_num = 68 #4
        path = f'/home/sciortino/SOLPS/RR_{shot}_attempts/Attempt{case_num}/Output/'
        gfilepath = '/home/sciortino/EFIT/g1120917011.00999_981'  # hard-coded
        form='extracted'
                 
        # load SOLPS results
        so = solps_case(path, gfilepath, case_num=case_num, form=form)
        
    elif case=='SPARC':
        path = '/home/sciortino/SPARC/V1E_LSN2_D+C'
        solps_run = 'P29MW_n1.65e20_Rout0.9_Y2pc'
        form='full'
        gfilepath = path+os.sep+'baserun'+os.sep+'V1E_geqdsk_LSN2'
        
        # load SOLPS results
        so = solps_case(path, gfilepath, solps_run=solps_run,form=form)
    else:
        raise ValueError('Unrecognized case')
    

    # fetch and process SOLPS output 
    so.process_solps_data(plot=False, edge_max=1.0) #0.05)
    quants = so.quants
    
    so.plot_2d_vals(np.log10(quants['NeuDen']), label=r'$n_n$')


    # compare SOLPS results at midplane with FSA
    rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS = so.get_n0_profiles()

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
    so.plot_2d_vals(np.log10(out['tot']*1e3), label=r'$log_{10}(P_{rad})$ [$kW/m^3$]')
    
    # plot total line radiated power
    so.plot_2d_vals(np.log10(out['line_rad'].sum(1)*1e3), label=r'$log_{10}(P_{line,rad})$ [$kW/m^3$]')

    if case=='CMOD':
        # overplot machine tiles
        overplot_machine(shot, [plt.gca()])
