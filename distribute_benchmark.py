import os
import sys
import time

import functools
import collections
import contextlib

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(r) for r in range(sys.argv[2])])

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import jax.numpy as jnp

import numpy as np
import h5py

from itertools import product
from tensorflow_probability.substrates import jax as tfp
from utils import array_maker
from runner import run_wrapper
from models import *

N_array,M_array,inds,vals = array_maker(*[int(s) for s in sys.argv[3:]])

model_type = sys.argv[1]
for packed in inds:
    i,j = [int(p) for p in packed.split(',')]
    N = N_array[i]
    M = M_array[j]

    dt_file = 'data/%s/%s_%s.hdf5'%(model_type,N,M)
    dt = h5py.File(dt_file,'a')

    def shard(data_array):
        split_data = [np.array(d).reshape((jax.device_count(),-1,*d.shape[1:])) for d in data_array]
        return(tuple([jax.pmap(lambda x: x)(s) for s in split_data]))

    sharded_data = shard([dt['data'][k] for k in dt['data_keys']])
    pass_data = sharded_data[-dt['n_comp'][0]:]

    run = run_wrapper(model_dict[model_type])

    start_time = time.perf_counter()
    states, trace = run(random.PRNGKey(0),sharded_data,pass_data)
    end_time = time.perf_counter()
    # print('%s,%s\t%s,%s'%(str(N),str(M),str(start_time),str(end_time)))
    # runtime = timeit('run(random.PRNGKey(0), (sharded_X, sharded_y))',number=1)

    n_param = len(dt['param_keys'])
    dt.create_dataset('runtime',data=float(end_time - start_time))
    for i in range(n_param):
        k = dt['param_keys'][i]
        dt['errors'].create_dataset(k,data=(states[i].mean(0) - dt['params'][k][()]))
