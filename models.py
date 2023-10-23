import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfed = tfp.experimental.distribute
tfde = tfp.experimental.distributions

def lmm_wrapper(X,y):
    n,m = X.shape
    Root = tfed.JointDistributionCoroutine.Root
    def model():
        s_beta = yield Root(tfd.Sample(tfd.HalfCauchy(0., 1.)))
        s_e = yield Root(tfd.Sample(tfd.HalfCauchy(0., 1.)))
        mu_beta = yield Root(tfd.Sample(tfd.Cauchy(0., 0.3)))
        beta = yield Root(tfd.Sample(tfd.MultivariateNormalDiag(
            mu_beta*jnp.ones(m), s_beta*jnp.ones(m)
        )))
        mu = jnp.dot(X, beta)
        yield tfed.Sharded(tfd.Independent(tfd.Normal(mu, s_e),
                                       reinterpreted_batch_ndims=1),
                                       shard_axis_name='data')
    return(model)

def am_wrapper(true_X,y):
    n,m = true_X.shape
    Root = tfed.JointDistributionCoroutine.Root
    def model():
        s_g = yield Root(tfd.HalfCauchy(0.,1.))
        s_e = yield Root(tfd.HalfCauchy(0.,1.))
        alpha = yield Root(tfd.HalfCauchy(0.,.2))
        beta = yield Root(tfd.MultivariateNormalDiag(jnp.zeros(m),s_g*jnp.ones(m)))
        X = yield Root(tfed.Sharded(tfd.Sample(tfd.MultivariateNormalDiagPlusLowRankCovariance(jnp.zeros(m),jnp.ones(m),jnp.sqrt(alpha)*beta[:,None]),sample_shape=n),shard_axis_name='data'))
        mu = jnp.dot(X,beta)
        yield tfed.Sharded(tfd.Independent(tfd.Normal(mu,s_e),reinterpreted_batch_ndims=1),shard_axis_name='data')

    return(model)

model_dict = {'lmm': lmm_wrapper, 'am': am_wrapper}
