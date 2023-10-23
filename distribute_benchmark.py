# !pip install seaborn --target=/kaggle/working/mysitepackages

import sys
import time

import functools
import collections
import contextlib

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import tensorflow_datasets as tfds

from itertools import product
from timeit import timeit
from tensorflow_probability.substrates import jax as tfp
from runner import run_wrapper
from models import *

from jax_smi import initialise_tracking
initialise_tracking()

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.mcmc
tfed = tfp.experimental.distribute
tfde = tfp.experimental.distributions
tfem = tfp.experimental.mcmc

Root = tfed.JointDistributionCoroutine.Root

min_N = int(sys.argv[2])
max_N = int(sys.argv[3])
N_t = (max_N - min_N)**2
min_M = int(sys.argv[4])
max_M = int(sys.argv[5])
M_t = (max_M - min_M)**2
N_array = np.concatenate([[10**i,5*10**i] for i in range(min_N,max_N)])
M_array = np.concatenate([[10**i,5*10**i] for i in range(min_M,max_M)])
t = max(N_t,M_t)

inds = np.fliplr(np.array(['%s,%s'%(p[0],p[1]) for p in product(np.arange(N_t),np.arange(M_t))]).reshape(N_t,M_t))
inds = np.concatenate([inds.diagonal(i) for i in range(t-1,-t,-1)])
vals = np.array(['%s,%s'%(p[0],p[1]) for p in product(N_array,M_array)]).reshape(N_t,M_t)

model_type = sys.argv[1]
for packed in inds:
    i,j = [int(p) for p in packed.split(',')]
    N = N_array[i]
    M = M_array[j]

    dt_file = 'data/%s/%s_%s.hdf5'%(model_type,N,M)
    dt = h5py.File(dt_file,'a')

    def shard(data_array):
        split_data = [np.array(d).reshape((jax.device_count(),-1,*d.shape[1:])) for d in data_array]
        return(tuple(split_data))

    sharded_data = shard([dt['data'][k] for k in dt['data_keys']])
    pass_data = sharded_data[-dt['n_comp'][0]:]

    run = run_wrapper(model_dict[model_type])

    start_time = time.perf_counter()
    states, trace = run(random.PRNGKey(0),sharded_data,pass_data)
    end_time = time.perf_counter()
    print('%s,%s\t%s,%s'%(str(N),str(M),str(start_time),str(end_time)))
    # runtime = timeit('run(random.PRNGKey(0), (sharded_X, sharded_y))',number=1)

    n_param = len(dt['param_keys'])
    dt.create_dataset('runtime',data=float(end_time - start_time))
    for i in range(n_param):
        k = dt['param_keys'][i]
        dt['errors'].create_dataset(k,data=(states[i].mean(0) - dt['params'][k][()]))
