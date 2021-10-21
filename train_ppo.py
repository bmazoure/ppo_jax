import os
from collections import deque

import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from absl import app, flags
from flax.training.train_state import TrainState
# from flax.training import checkpoints
from jax.random import PRNGKey

from algo import get_transition, select_action, update
from buffer import Batch
from models import TwinHeadModel
from vec_env import ProcgenVecEnvCustom


def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("num_envs", 64, "Num of Procgen envs.")
flags.DEFINE_integer("train_steps", 25_000_000, "Number of train frames.")
# PPO
flags.DEFINE_float("max_grad_norm", 0.5, "Max grad norm")
flags.DEFINE_float("gamma", 0.999, "Gamma")
flags.DEFINE_integer("n_steps", 256, "GAE n-steps")
flags.DEFINE_integer("n_minibatch", 8, "Number of PPO minibatches")
flags.DEFINE_float("lr", 5e-4, "PPO learning rate")
flags.DEFINE_integer("epoch_ppo", 3, "Number of PPO epochs on a single batch")
flags.DEFINE_float("clip_eps", 0.2, "Clipping range")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda")
flags.DEFINE_float("entropy_coeff", 0.01, "Entropy loss coefficient")
flags.DEFINE_float("critic_coeff", 0.5, "Value loss coefficient")
# Logging
flags.DEFINE_integer("checkpoint_interval", 999424, "Chcekpoint frequency (about 1M)")
flags.DEFINE_string("model_dir", "model_weights", "Model weights dir")
flags.DEFINE_string("run_id", "jax_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_boolean("disable_wandb", True, "Disable W&B logging")


def main(argv):
    # Clip rewards for training
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                              num_levels=200,
                              mode='easy',
                              start_level=0,
                              paint_vel_info=False,
                              num_envs=FLAGS.num_envs,
                              normalize_rewards=True)
    # Report unclipped rewards for test
    env_test_ID = ProcgenVecEnvCustom(FLAGS.env_name,
                                      num_levels=200,
                                      mode='easy',
                                      start_level=0,
                                      paint_vel_info=False,
                                      num_envs=FLAGS.num_envs,
                                      normalize_rewards=False)
    env_test_OOD = ProcgenVecEnvCustom(FLAGS.env_name,
                                       num_levels=0,
                                       mode='easy',
                                       start_level=0,
                                       paint_vel_info=True,
                                       num_envs=FLAGS.num_envs,
                                       normalize_rewards=False)
                                       
    os.environ["WANDB_API_KEY"] = "API_KEY"
    group_name = "%s_%s" % (FLAGS.env_name, FLAGS.run_id)
    name = "jax_ppo_%s_%s_%d" % (FLAGS.env_name, FLAGS.run_id,
                                        np.random.randint(100000000))

    wandb.init(project='project', entity='username', config=FLAGS, group=group_name, name=name, sync_tensorboard=True, mode="disabled" if \
               FLAGS.disable_wandb else "online")
    
    np.random.seed(FLAGS.seed)
    key = PRNGKey(FLAGS.seed)

    model = TwinHeadModel(action_dim=env.action_space.n,
                          prefix_critic='vfunction',
                          prefix_actor="policy")
    fake_args_model = jnp.zeros((1, 64, 64, 3))
    params_model = model.init(key, fake_args_model)

    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.max_grad_norm),
        optax.adam(FLAGS.lr, eps=1e-5)
    )
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params_model,
        tx=tx)

    batch = Batch(
            discount=FLAGS.gamma,
            gae_lambda=FLAGS.gae_lambda,
            n_steps=FLAGS.n_steps+1,
            num_envs=FLAGS.num_envs,
            state_space=env.observation_space
        )
        
    state = env.reset()
    state_id = env_test_ID.reset()
    state_ood = env_test_OOD.reset()

    epinfo_buf_id = deque(maxlen=100)
    epinfo_buf_ood = deque(maxlen=100)

    model_dir = os.path.join(FLAGS.model_dir, FLAGS.env_name)

    for step in tqdm.tqdm(range(1, int(FLAGS.train_steps // FLAGS.num_envs + 1))):
        train_state, state, batch, key = get_transition(train_state, env, state, batch, key)

        action_id, _, _, key = select_action(train_state, state_id.astype(jnp.float32) / 255., key, sample=True)
        state_id, _, _, infos_id = env_test_ID.step(action_id)

        action_ood, _, _, key = select_action(train_state, state_ood.astype(jnp.float32) / 255., key, sample=True)
        state_ood, _, _, infos_ood = env_test_OOD.step(action_ood)

        for info in infos_id:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_id.append(maybe_epinfo)

        for info in infos_ood:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_ood.append(maybe_epinfo)

        if step % (FLAGS.n_steps + 1) == 0:
            
            metric_dict, train_state, key = update(train_state,
                                                batch.get(),
                                                FLAGS.num_envs,
                                                FLAGS.n_steps,
                                                FLAGS.n_minibatch,
                                                FLAGS.epoch_ppo,
                                                FLAGS.clip_eps,
                                                FLAGS.entropy_coeff,
                                                FLAGS.critic_coeff,
                                                key)
            batch.reset()
            renamed_dict = {}
            for k,v in metric_dict.items():
                renamed_dict["%s/%s"%(FLAGS.env_name,k)] = v
            wandb.log(
                renamed_dict,
                step=FLAGS.num_envs * step)

            wandb.log({
                "%s/ep_return_200" % (FLAGS.env_name): safe_mean([info['r'] for info in epinfo_buf_id]),
                "step": FLAGS.num_envs * step
            })
            wandb.log({
                "%s/ep_return_all" % (FLAGS.env_name): safe_mean([info['r'] for info in epinfo_buf_ood]),
                "step": FLAGS.num_envs * step
            })
            print('Eprew: %.3f'%safe_mean([info['r'] for info in epinfo_buf_id]))
            print("Logits L1 norm: %.3f" % jnp.linalg.norm(train_state.params['params']['policy/fc_pi']['kernel']) )

        if ( step * FLAGS.num_envs ) % FLAGS.checkpoint_interval == 0:
            print('Saving model weights [skipping for now]')
            # checkpoints.save_checkpoint('./%s'%model_dir, train_state, step * FLAGS.num_envs)

if __name__ == '__main__':
    app.run(main)
    quit()