import sys
import h5py
import numpy as np

from utils import array_maker

N_array,M_array,inds,vals = array_maker(*[int(s) for s in sys.argv[3:]])

runtimes = []
model_type = sys.argv[1]
for packed in inds:
    i,j = [int(p) for p in packed.split(',')]
    N = N_array[i]
    M = M_array[j]

    dt_file = 'data/%s/%s_%s.hdf5'%(model_type,N,M)
    dt = h5py.File(dt_file,'r')

    runtimes.append(dt['runtime'][()])

np.savetxt('data/%s/%sGPU_%s_%s_%s_%s.txt'%(tuple(sys.argv[1:])),np.array(runtimes))
