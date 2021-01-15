'''
File names:
fort.33, fort.34, fort.35: EIRENE output mesh  (nodes, cells, [neighbors])
b2fgmtry: B2 mesh
mesh.extra: vessel geometry
b2fstate: B2 output
fort44, fort46: EIRENE output, on curvilinear grid of B2 and triangular mesh of EIRENE, respectively

In fort44 (B2 grid):
dab2 = neutral atomic density
tab2 = neutral atomic temperature

In fort46 (EIRENE grid) -- see p.436 of SOLPS-ITER-2020 manual
pdena = neutral atomic density
pdenm = neutral molecular density

fort 34: index of every triangle in the EIRENE mesh
'''


import omfit_solps, omfit_eqdsk
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import griddata,RectBivariateSpline,interp2d
from matplotlib import ticker, cm
import matplotlib.tri as tri
from matplotlib.pyplot import tripcolor

from solps_names import solps_definitions
definitions = solps_definitions()

baserun = '/home/sciortino/SPARC/V1E_LSN2_D+C/baserun/'
case = '/home/sciortino/SPARC/V1E_LSN2_D+C/P29MW_n1.65e20_Rout0.9_Y2pc/'

b2fstate = omfit_solps.OMFITsolps(case+'b2fstate')
geom = omfit_solps.OMFITsolps(baserun+'b2fgmtry')

# not read-in correctly... how to? 
#mesh = omfit_solps.OMFITsolps(baserun+'mesh.extra')

# 190*72*2 = 27360
def load_eirene_output(filepath = case+'fort.46'):
    with open(filepath, 'r') as f:
        contents = f.readlines()
    ii=6
    vals = {}
    while ii<len(contents[ii:]):
        if  contents[ii].startswith('*eirene'):
            key = contents[ii].split()[3]
            vals[key] = []
        else:
            [vals[key].append(float(val)) for val in contents[ii].strip().split()]
        ii+=1
    return vals

fort44 = load_eirene_output(filepath = case+'fort.44')
fort46 = load_eirene_output(filepath = case+'fort.46')

for key in fort46:
    fort46[key] = np.array(fort46[key])
for key in fort44:
    fort44[key] = np.array(fort44[key])

# read EIRENE nodes and cells
Nodes=np.fromfile(baserun+'fort.33',sep=' ')
NN=int(Nodes[0])
XNodes=Nodes[1:NN+1]/100  # cm -->m
YNodes=Nodes[NN+1:]/100

# simple to read triangles:
Triangles=np.loadtxt(baserun+'fort.34',skiprows=1, usecols=(1,2,3))

TP=tri.Triangulation(XNodes,YNodes,triangles=(Triangles-1))

# Molecular D?
fig,ax = plt.subplots(figsize=(4,7))
#ax.plot(VVFILE[:,0]/1000,VVFILE[:,1]/1000,'k-')  # from mesh.extra?
IM=ax.tripcolor(TP, fort46['pdenm'],shading='flat')   # cm^-3 # manual p.151
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('Radial Position R (m)')
ax.set_ylabel('Vertical Position Z (m)')
plt.colorbar(IM,ax=ax)

## D #1 species ?
fig,ax = plt.subplots(figsize=(4,7))
IM=ax.tripcolor(TP, np.log10(fort46['pdena'][:fort46['pdena'].shape[0]//2]))
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('Radial Position R (m)')
ax.set_ylabel('Vertical Position Z (m)')
plt.colorbar(IM,ax=ax)

## D#2 species ?
fig,ax = plt.subplots(figsize=(4,7))
IM=ax.tripcolor(TP, fort46['pdena'][fort46['pdena'].shape[0]//2:])
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('Radial Position R (m)')
ax.set_ylabel('Vertical Position Z (m)')
plt.colorbar(IM,ax=ax)





##### Try loading methods similar to those of J. Lore
# number of species:
ns = b2fstate['ns']   # 2 for pure plasma

# load geqdsk
geqdsk = omfit_eqdsk.OMFITgeqdsk(baserun+'V1E_geqdsk_LSN2')


# magnetic field flux functions
#fpsi = b2fstate['fpsi']
#ffbz = b2fstate['ffbz']

# toroidal and cylindrical symmetry:
# bx=(dpsi/dy)/(2*pi*R)
# by=-(dpsi/dx)/(2*pi*R)
# bz=ffbz/(2*pi*R)

###
nx,ny = geom['nx,ny']
crx = geom['crx'].reshape(4,ny+2,nx+2)
cry = geom['cry'].reshape(4,ny+2,nx+2)

# (,,0): lower left corner, (,,1): lower right corner
# (,,2): upper left corner, (,,3): upper right corner.

# plot location of grid points
fig,ax = plt.subplots()
ax.plot(np.mean(crx,axis=0), np.mean(cry,axis=0),'.')
ax.axis('equal')

# create triangulation at mesh midpoints
x = np.mean(crx,axis=0)
y = np.mean(cry,axis=0)

fig,ax = plt.subplots()
plt.contourf(x,y,b2fstate['te'])
ax.axis('equal')


####
import aurora_solps
triang = aurora_solps.apply_mask(np.mean(crx,axis=2), np.mean(cry,axis=2),
           geqdsk, dist=0.5, mask_up=False, mask_down=False)
fig,ax = plt.subplots()
ax.tricontourf(triang, np.log10(b2fstate['ne']).flatten())#,cmap=cm.magma,levels=300)
ax.axis('equal')
