from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optimizers
from jax.tree_util import tree_flatten

@partial(jax.jit, static_argnums=(0, 1, 4))
def optimize(
    fn_loss: Any,
    opt: Any,
    opt_state: Any,
    max_grad_norm: float or None,
    params: Any,
    step: int,
    *args,
    **kwargs,
) -> Tuple[Any, jnp.ndarray, Any]:
    (loss, aux), grad = jax.value_and_grad(fn_loss, has_aux=True)(
        params,
        *args,
        **kwargs,
    )
    if max_grad_norm is not None:
        grad = clip_gradient_norm(grad, max_grad_norm)
    opt_state = opt(step, grad, opt_state)
    
    return opt_state, loss, aux

def fake_state(state_space):
    state = state_space.sample()[None, ...]
    if len(state_space.shape) == 1:
        state = state.astype(np.float32)
    return 

@jax.jit
def clip_gradient(
    grad: Any,
    max_value: float,
) -> Any:
    """
    Clip gradients.
    """
    return jax.tree_map(lambda g: jnp.clip(g, -max_value, max_value), grad)


@jax.jit
def clip_gradient_norm(
    grad: Any,
    max_grad_norm: float,
) -> Any:
    """
    Clip norms of gradients.
    """

    def _clip_gradient_norm(g):
        clip_coef = max_grad_norm / (jax.lax.stop_gradient(jnp.linalg.norm(g)) + 1e-6)
        clip_coef = jnp.clip(clip_coef, a_max=1.0)
        return g * clip_coef

    return jax.tree_map(lambda g: _clip_gradient_norm(g), grad)


@jax.jit
def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
    """
    Update target network using Polyak-Ruppert Averaging.
    """
    return jax.tree_multimap(lambda t, s: (1 - tau) * t + tau * s, target_params, online_params)


@jax.jit
def weight_decay(params: hk.Params) -> jnp.ndarray:
    """
    Calculate the sum of L2 norms of parameters.
    """
    leaves, _ = tree_flatten(params)
    return 0.5 * sum(jnp.vdot(x, x) for x in leaves)

def replace_params(filter_s,filter_t,params_s,params_t):
    selected_params_s = hk.data_structures.filter(filter_s, params_s)
    selected_params_t = hk.data_structures.filter(filter_t, params_t)
    i = 0
    for module_name, name, value in hk.data_structures.traverse(selected_params_t):
        if i > 0:
            continue
        i += 1
    template_name = '/'.join(module_name.split('/')[:2])
    for module_name, name, value in hk.data_structures.traverse(selected_params_s):
        new_name='/'.join(module_name.split('/')[2:])
        new_name = template_name + '/' + new_name
        params_t[new_name]=value
        
    return params_t #hk.data_structures.to_immutable_dict(params_t)

def evaluate(env, select_action, train_state, num_episodes):
    total_return = 0.0
    eps_count = 0
    state = env.reset()
    while eps_count < num_episodes:
        action = select_action(train_state, state.astype(jnp.float32)/255.)
        state, reward, done, info = env.step(action)
        for inf in info:
            if 'episode' in inf:
                ep_ret = inf['episode']['r']
                total_return += ep_ret
        eps_count += np.sum(done)
    mean_return = total_return / num_episodes
    return mean_return