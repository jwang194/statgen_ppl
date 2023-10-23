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

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.mcmc
tfed = tfp.experimental.distribute
tfde = tfp.experimental.distributions
tfem = tfp.experimental.mcmc

Root = tfed.JointDistributionCoroutine.Root

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
