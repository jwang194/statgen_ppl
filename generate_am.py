import sys
import functools
import collections
from itertools import product

import numpy as np
import h5py

min_N = int(sys.argv[1])
max_N = int(sys.argv[2])
N_t = (max_N - min_N)*2
min_M = int(sys.argv[3])
max_M = int(sys.argv[4])
M_t = (max_M - min_M)*2
N_array = np.concatenate([[10**i,5*10**i] for i in range(min_N,max_N)])
M_array = np.concatenate([[10**i,5*10**i] for i in range(min_M,max_M)])
t = max(N_t,M_t)

inds = np.fliplr(np.array(['%s,%s'%(p[0],p[1]) for p in product(np.arange(N_t),np.arange(M_t))]).reshape(N_t,M_t))
inds = np.concatenate([inds.diagonal(i) for i in range(t-1,-t,-1)])
vals = np.array(['%s,%s'%(p[0],p[1]) for p in product(N_array,M_array)]).reshape(N_t,M_t)

for packed in inds:
    i,j = [int(p) for p in packed.split(',')]
    N = N_array[i]
    M = M_array[j]
    print('%s_%s'%(N,M))

    dt = h5py.File('data/am/%s_%s.hdf5'%(N,M),'w')

    s_g = np.sqrt(np.random.uniform(0,1))
    s_e = np.sqrt(1-s_g**2)
    alpha = np.random.uniform(0,.3)
    beta = np.random.randn(M)*s_g

    X = np.random.multivariate_normal(mean=np.zeros(M),cov=np.eye(M)+alpha*np.outer(beta,beta),size=N)
    e = np.random.randn(N)*s_e
    y = X @ beta + e

    dt.create_group('params')
    dt.create_group('data')
    dt.create_group('errors')
    param_keys = ('s_g','s_e','alpha','beta')
    dt.create_dataset('param_keys',data=param_keys)
    data_keys = ('X','y')
    dt.create_dataset('data_keys',data=data_keys)
    dt.create_dataset('n_comp',data=[2])
    
    for p,pk in zip((s_g,s_e,alpha,beta),param_keys):
        dt['params'].create_dataset(pk,data=np.array(p))

    for d,dk in zip((X,y),data_keys):
        dt['data'].create_dataset(dk,data=d)

