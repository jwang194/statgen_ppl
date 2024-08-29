import numpy as np
from itertools import product

def rounder(N_GPU,min_N,max_N):
    return np.concatenate([[N_GPU * round((10**i)/N_GPU), N_GPU * round((5*10**i)/N_GPU)] for i in range(min_N,max_N)])

def array_maker(N_GPU,min_N,max_N,min_M,max_M):
    N_t = (max_N - min_N)*2
    M_t = (max_M - min_M)*2
    N_array = rounder(N_GPU,min_N,max_N)
    M_array = np.concatenate([[10**i,5*10**i] for i in range(min_M,max_M)])
    t = max(N_t,M_t)

    inds = np.fliplr(np.array(['%s,%s'%(p[0],p[1]) for p in product(np.arange(N_t),np.arange(M_t))]).reshape(N_t,M_t))
    inds = np.concatenate([inds.diagonal(i) for i in range(t-1,-t,-1)])
    vals = np.array(['%s,%s'%(p[0],p[1]) for p in product(N_array,M_array)]).reshape(N_t,M_t)

    return((N_array,M_array,inds,vals))
