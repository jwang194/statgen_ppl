# !pip install seaborn --target=/kaggle/working/mysitepackages
import sys
import os

import functools
import collections
import contextlib

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import jax.numpy as jnp

import numpy as np

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.internal.structural_tuple import structtuple

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.mcmc
tfed = tfp.experimental.distribute
tfde = tfp.experimental.distributions
tfem = tfp.experimental.mcmc

Root = tfed.JointDistributionCoroutine.Root

N_RESULT = 10**3
N_BURNIN = 10**3

def run_wrapper(model_wrapper):
    # set up the sharded markov chain
    @functools.partial(jax.pmap, axis_name='data', in_axes=(None, 0, 0), out_axes=None)
    # @jax.default_matmul_precision('tensorfloat32')
    def run(seed, data, pass_data):
        model_fn = model_wrapper(*data)
        model = tfed.JointDistributionCoroutine(model_fn)

        init_seed, sample_seed = random.split(seed)

        n_comp = len(pass_data)
        initial_state = model.sample(seed=init_seed)[:-n_comp] # throw away `y`

        def target_log_prob(*params):
            return model.log_prob(params + pass_data)

        kernel = tfp.mcmc.NoUTurnSampler(target_log_prob, 1e-3)

        states, trace = tfm.sample_chain(num_results=1000,
                              current_state=initial_state,
                              kernel=kernel,
                              trace_fn=lambda _,
                              results: results.target_log_prob,
                              num_burnin_steps=1000,
                              seed=sample_seed)
        return states, trace

    return(run)

def smart_run_wrapper(model_name,model_wrapper,raw_data,N_BURNIN=N_BURNIN,N_RESULT=N_RESULT):
    if model_name == 'lmm':
        struct_tuple = structtuple(['var0','var1','var2','var3'])
        X,y = raw_data
        M = X.shape[1]
        beta_estimates = jnp.array([(X[:,i].T @ X[:,i])**(-1) * X[:,i].T @ y for i in range(M)])
        initial_state = struct_tuple(
                beta_estimates.std(),
                jnp.sqrt(1-beta_estimates.var()),
                beta_estimates.mean(),
                beta_estimates)
    elif model_name == 'spsl':
        struct_tuple = structtuple(['var0','var1','var2','var3'])
        X,y = data
        beta_estimates = jnp.array([(X[:,i].T @ X[:,i])**(-1) * X[:,i].T @ y for i in range(M)])
        initial_state = initial_state._replace(
                0.2,
                beta_estimates.std(),
                jnp.sqrt(1-beta_estimates.var()),
                beta_estimates)
    # set up the sharded markov chain
    @functools.partial(jax.pmap, axis_name='data', in_axes=(None, 0, 0), out_axes=None)
    # @jax.default_matmul_precision('tensorfloat32')
    def run(seed, data, pass_data):
        model_fn = model_wrapper(*data)
        model = tfed.JointDistributionCoroutine(model_fn)

        init_seed, sample_seed = random.split(seed)

        n_comp = len(pass_data)
        if model_name in ['lmm','spsl']:
            initial_state = model.sample(seed=init_seed)[:-n_comp] # throw away `y`

        def target_log_prob(*params):
            return model.log_prob(params + pass_data)

        kernel = tfp.mcmc.NoUTurnSampler(target_log_prob, 1e-3)

        states, trace = tfm.sample_chain(num_results=N_RESULT,
                              current_state=initial_state,
                              kernel=kernel,
                              trace_fn=lambda _,
                              results: results.target_log_prob,
                              num_burnin_steps=N_BURNIN,
                              seed=sample_seed)
        return states, trace

    return(run)
