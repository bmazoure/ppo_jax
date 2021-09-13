from absl import flags
from absl import app
import os
import numpy as np
import tqdm
import jax.numpy as jnp
import time
from tensorboardX import SummaryWriter

from env_wrappers import ProcgenVecEnvCustom
from algo import get_transition, update, select_action
from utils import evaluate

from jax.random import PRNGKey
from models import TwinHeadModel
from flax.training.train_state import TrainState
import optax

from buffer import Batch

import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 123, "Random seed.")
flags.DEFINE_integer("num_envs", 64, "Num of Procgen envs.")
flags.DEFINE_integer("train_steps", 25_000_000, "Number of train frames.")
flags.DEFINE_integer("num_eval_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 1000,
                     "Number of episodes between evaluations.")
flags.DEFINE_float("max_grad_norm", 0.5, "Max grad norm")
flags.DEFINE_float("gamma", 0.999, "Gamma")
flags.DEFINE_integer("n_steps", 256, "GAE n-steps")
flags.DEFINE_integer("n_minibatch", 8, "Number of PPO minibatches")
flags.DEFINE_float("lr", 5e-4, "PPO learning rate")
flags.DEFINE_integer("epoch_ppo", 3, "Number of PPO epochs on a single batch")
flags.DEFINE_float("clip_eps", 0.2, "Clipping range")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda")
flags.DEFINE_float("entropy_coeff", 0.01, "Entropy loss coefficient")
flags.DEFINE_float("critic_coeff", 1, "Value loss coefficient")
flags.DEFINE_string("run_id", "jax_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_boolean("disable_wandb", True, "Disable W&B logging")


def main(argv):
    # Initialize the environment.
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                              num_levels=200,
                              mode='easy',
                              start_level=0,
                              paint_vel_info=True,
                              num_envs=FLAGS.num_envs)
    env_test_ID = ProcgenVecEnvCustom(FLAGS.env_name,
                                      num_levels=200,
                                      mode='easy',
                                      start_level=0,
                                      paint_vel_info=True,
                                      num_envs=FLAGS.num_envs)
    env_test_OOD = ProcgenVecEnvCustom(FLAGS.env_name,
                                       num_levels=0,
                                       mode='easy',
                                       start_level=0,
                                       paint_vel_info=True,
                                       num_envs=FLAGS.num_envs)

    os.environ["WANDB_API_KEY"] = "KEY"
    group_name = "%s_%s" % (FLAGS.env_name, FLAGS.run_id)
    name = "shared_encoder_%s_%s_%d" % (FLAGS.env_name, FLAGS.run_id,
                                        np.random.randint(100000000))

    wandb.init(project='project', entity='entity', config=FLAGS, group=group_name, name=name, mode="disabled" if \
               FLAGS.disable_wandb else "online")
    
    np.random.seed(FLAGS.seed)
    key = PRNGKey(FLAGS.seed)

    model = TwinHeadModel(action_dim=env.action_space.n,
                          prefix_critic='vfunction',
                          prefix_actor="policy")
    fake_args_model = jnp.zeros((1, 64, 64, 3))
    params_model = model.init(key, fake_args_model)

    tx = optax.chain(
        optax.adam(FLAGS.lr),
        optax.clip_by_global_norm(FLAGS.max_grad_norm)
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

    for step in tqdm.tqdm(
            range(1, int(FLAGS.train_steps // FLAGS.num_envs + 1))):

        train_state, state, batch, key = get_transition(train_state, env, state, batch, key)
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

        if (step * FLAGS.num_envs) % FLAGS.evaluate_every == 0:
            """
            Evaluate the PPO agent on the test_env
            """
            eval_policy = lambda x,y: select_action(x,y,key,sample=False)[0]
            eval_returns_ID = evaluate(env_test_ID, eval_policy, train_state,
                                       FLAGS.num_eval_episodes)
            eval_returns_OOD = evaluate(env_test_OOD, eval_policy, train_state,
                                        FLAGS.num_eval_episodes)
            wandb.log({
                "%s/ep_return_200" % (FLAGS.env_name): eval_returns_ID,
                "step": FLAGS.num_envs * step
            })
            wandb.log({
                "%s/ep_return_all" % (FLAGS.env_name): eval_returns_OOD,
                "step": FLAGS.num_envs * step
            })


if __name__ == '__main__':
    app.run(main)
