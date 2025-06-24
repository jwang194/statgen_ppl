import os
import sys
import time

import functools
import collections
import contextlib

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(r) for r in range(int(sys.argv[2]))])

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
from runner import run_wrapper,smart_run_wrapper
from models import *
from generate_lmm import *
# from generate_am import *
from generate_spsl import *

N_GPU = int(sys.argv[2])

N,M = [int(v) for v in sys.argv[3:-2]]

scale = sys.argv[-2] == 'True'
smart_init = sys.argv[-1] == 'True'

model_type = sys.argv[1]

globals()['generate_%s'%model_type](N,M,scale,smart_init)

dt_file = 'data/%s/%s_%s%s.hdf5'%(model_type,N,M,'_scaled' if scale else '')
dt = h5py.File(dt_file,'a')

def shard(data_array):
    split_data = [np.array(d).reshape((jax.device_count(),-1,*d.shape[1:])) for d in data_array]
    return(tuple([jax.pmap(lambda x: x)(s) for s in split_data]))

raw_data = [dt['data'][k][()] for k in dt['data_keys']]
sharded_data = shard(raw_data)
pass_data = sharded_data[-dt['n_comp'][0]:]

if smart_init:
    run = smart_run_wrapper(model_type,model_dict[model_type],raw_data)
else:
    run = run_wrapper(model_type,model_dict[model_type])

start_time = time.perf_counter()
states, trace = run(random.key(0),sharded_data,pass_data)
end_time = time.perf_counter()
print('%s,%s\t%s-%s'%(str(N),str(M),str(start_time),str(end_time)))
# runtime = timeit('run(random.PRNGKey(0), (sharded_X, sharded_y))',number=1)

n_param = len(dt['param_keys'])
smart_init_tag = '_smart' if smart_init else ''
if 'runtime_%i_GPU%s'%(N_GPU,smart_init_tag) in dt:
    del dt['runtime_%i_GPU%s'%(N_GPU,smart_init_tag)]
    dt.create_dataset('runtime_%i_GPU%s'%(N_GPU,smart_init_tag),data=float(end_time - start_time))
else:
    dt.create_dataset('runtime_%i_GPU%s'%(N_GPU,smart_init_tag),data=float(end_time - start_time))
for i in range(n_param):
    k = dt['param_keys'][i]
    if 'errors/%s_%i_GPU%s'%(k,N_GPU,smart_init_tag) in dt:
        del dt['errors']['%s_%i_GPU%s'%(k,N_GPU,smart_init_tag)]
        dt['errors'].create_dataset('%s_%i_GPU%s'%(k,N_GPU,smart_init_tag),data=(states[i].mean(0) - dt['params'][k][()]))
    else:
        dt['errors'].create_dataset('%s_%i_GPU%s'%(k,N_GPU,smart_init_tag),data=(states[i].mean(0) - dt['params'][k][()]))
    if 'states/%s_%i_GPU%s'%(k,N_GPU,smart_init_tag) in dt:
        del dt['states']['%s_%i_GPU%s'%(k,N_GPU,smart_init_tag)]
        dt['states'].create_dataset('%s_%i_GPU%s'%(k,N_GPU,smart_init_tag),data=states[i])
    else:
        dt['states'].create_dataset('%s_%i_GPU%s'%(k,N_GPU,smart_init_tag),data=states[i])
