import os
import sys
import functools
import collections
from itertools import product

import numpy as np
import h5py

from utils import array_maker

N_array,M_array,inds,vals = array_maker(*[int(s) for s in sys.argv[1:]])

for packed in inds:
    i,j = [int(p) for p in packed.split(',')]
    N = N_array[i]
    M = M_array[j]

    if os.path.isfile('data/lmm/%s_%s.hdf5'%(N,M)):
        continue

    dt = h5py.File('data/lmm/%s_%s.hdf5'%(N,M),'w')

    s_g = np.sqrt(np.random.uniform(0,1))
    s_e = np.sqrt(1-s_g**2)
    mu_beta = np.random.uniform(0,.3)
    beta = np.random.randn(M)*s_g + mu_beta

    X = np.random.randn(M*N).reshape(N,M)
    X = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x), 0, X) / np.sqrt(M)
    e = np.random.randn(N)*s_e
    y = X @ beta + e
    y -= np.mean(y)

    dt.create_group('params')
    dt.create_group('data')
    dt.create_group('errors')
    param_keys = ('s_g','s_e','mu_beta','beta')
    dt.create_dataset('param_keys',data=param_keys)
    data_keys = ('X','y')
    dt.create_dataset('data_keys',data=data_keys)
    dt.create_dataset('n_comp',data=[1])
    
    for p,pk in zip((s_g,s_e,mu_beta,beta),param_keys):
        dt['params'].create_dataset(pk,data=np.array(p))

    for d,dk in zip((X,y),data_keys):
        dt['data'].create_dataset(dk,data=d)

