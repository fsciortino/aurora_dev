import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy, os, math
import scipy.interpolate
plt.ion()
from scipy.interpolate import LinearNDInterpolator


def load_adf12(file):
    f = open(file,'r')

    nlines = int(f.readline())

    cer_data = []
    for iline in np.arange(nlines):
        cer_line = {}
        params = []
        first_line = '0'
        while (not first_line[0].isalpha()):
            first_line =  f.readline()
        
        cer_line['header'] = first_line
        cer_line['qefref'] = np.float(f.readline()[:63].replace('D', 'e'))
        cer_line['parmref'] = np.float_(f.readline()[:63].replace('D', 'e').split())
        cer_line['nparmsc'] = np.int_(f.readline()[:63].split())
        
        for ipar, npar in enumerate(cer_line['nparmsc']):
            for q in np.arange(2):
                data = []                    
                while npar > len(data):
                    line = f.readline()
                    if len(line) > 63: 
                        name = line[63:].strip().lower()
                        cer_line[name] = []
                        if q == 0: params.append(name)
                    
                    values = np.float_(line[:63].replace('D', 'E').split())
                    values = values[values > 0]
                    if not len(values):
                        continue
                    data += values.tolist()
                cer_line[name] = data
        cer_data.append(cer_line)
        

    f.close()
    return params,cer_data



def interpolate_data(params,cer_line, n_par = 3):
    X = np.zeros(( np.sum(cer_line['nparmsc'][:n_par]),n_par))
    Y = np.zeros(( np.sum(cer_line['nparmsc'][:n_par])))

    for i in np.arange(n_par):
        X[:,i] = cer_line['parmref'][i]
        
    for i in np.arange(n_par):
        X[np.sum(cer_line['nparmsc'][:i]):np.sum(cer_line['nparmsc'][:i+1]),i] = cer_line[params[i]]
        Y[np.sum(cer_line['nparmsc'][:i]):np.sum(cer_line['nparmsc'][:i+1]) ]  = cer_line['q'+params[i]]
            
    interp = LinearNDInterpolator(np.log(X),np.log(Y), fill_value=0,rescale=True)

    return lambda xi: np.exp(interp(np.log(xi)))
    

if __name__=='__main__':

    fileloc='/fusion/projects/toolbox/sciortinof/atomlib/atomdat_master/adf12/qef93#h_c6.dat'

    paramsC6,cer_dataC6 = load_adf12(fileloc)

    E_eV = 70e3
    m_beam = 2.0
    Ti_eV = 3e3
    ne_cm3 = 1e14
    Zeff = 2.0
    rate = interpolate_data(paramsC6,cer_dataC6[4],n_par=4)(
        (E_eV/m_beam, Ti_eV, ne_cm3, Zeff)
        )
