import sys
import h5py
import numpy as np

from utils import array_maker

N_GPU = int(sys.argv[2])
N_array,M_array,inds,vals = array_maker(*[int(s) for s in sys.argv[2:]])

runtimes = []
model_type = sys.argv[1]
for packed in inds:
    i,j = [int(p) for p in packed.split(',')]
    N = N_array[i]
    M = M_array[j]

    dt_file = 'data/%s/%s_%s.hdf5'%(model_type,N,M)
    dt = h5py.File(dt_file,'r')

    runtimes.append((N,M,dt['runtime_%i_GPU'%N_GPU][()]))

runtimes = np.array(runtimes)
np.savetxt('data/%s/%sGPU_%s_%s_%s_%s.txt'%(tuple(sys.argv[1:])),runtimes,'%s')
