import os
from collections import deque
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax.training.train_state import TrainState
from flax.training import checkpoints
from jax.random import PRNGKey

from models import TwinHeadModel
from vec_env import ProcgenVecEnvCustom
import jax

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("num_envs", 64, "Num of Procgen envs.")
# PPO
flags.DEFINE_float("max_grad_norm", 0.5, "Max grad norm")
flags.DEFINE_float("lr", 5e-4, "PPO learning rate")
# Logging
flags.DEFINE_string("model_dir", "model_weights", "Model weights dir")

def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)

def main(argv):
    # Report unclipped rewards for test
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                                        num_levels=200,
                                        mode='easy',
                                        start_level=0,
                                        paint_vel_info=False,
                                        num_envs=FLAGS.num_envs,
                                        normalize_rewards=False)
                                        
    np.random.seed(FLAGS.seed)
    rng = PRNGKey(FLAGS.seed)

    model = TwinHeadModel(action_dim=env.action_space.n,
                            prefix_critic='vfunction',
                            prefix_actor="policy")
    fake_args_model = jnp.zeros((1, 64, 64, 3))
    params_model = model.init(rng, fake_args_model)
    
    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.max_grad_norm),
        optax.adam(FLAGS.lr, eps=1e-5)
    )
    
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params_model,
        tx=tx)
        
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.env_name)
    loaded_state = checkpoints.restore_checkpoint('./%s'%model_dir,target=train_state)

    @partial(jax.jit, static_argnums=0)
    def policy(apply_fn, params, state):
        value, logits = apply_fn(params, state)
        return value, logits

    def select_action(train_state, state,rng, greedy=False):
        state = state.astype(jnp.float32) / 255.
        value, logits = policy(train_state.apply_fn, train_state.params, state)
        rng, key = jax.random.split(rng)
        if greedy:
            action = logits.argmax(1)
        else:
            action = jax.random.categorical(key, logits)
        return action, value[:, 0], rng

    epinfo_buf = deque(maxlen=100)

    state = env.reset()
    done = np.array([False]*FLAGS.num_envs)
    action_seq, value_seq, level_seq = [], [], []
    obs_seq = []
    while not np.all(done):
        action, value, rng = select_action(loaded_state, state, rng, greedy=True)
        
        state, reward, _, epinfo = env.step(action)
        
        for i,info in enumerate(epinfo):
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf.append(maybe_epinfo)
                done[i] = True

    returns = safe_mean([info['r'] for info in epinfo_buf])

    print('Returns [%s]: %.3f'% (FLAGS.env_name,returns))

if __name__ == '__main__':
    app.run(main)
    quit()