from absl import flags
from absl import app
import os
import numpy as np
import tqdm
import jax.numpy as jnp
import time
import pandas as pd
from tensorboardX import SummaryWriter

from env_wrappers import ProcgenVecEnvCustom
from algo import PPO
from utils import evaluate

import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 123, "Random seed.")
flags.DEFINE_integer("num_envs", 64, "Num of Procgen envs.")
flags.DEFINE_integer("train_steps", 25_000_000, "Number of train frames.")
flags.DEFINE_integer("num_eval_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 1000,
                     "Number of episodes between evaluations.")
flags.DEFINE_string("run_id", "jax_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_boolean("disable_wandb", True, "Disable W&B logging")


def main(argv):
    # Initialize the environment.
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                              num_levels=200,
                              mode='easy',
                              start_level=0,
                              paint_vel_info=False,
                              num_envs=FLAGS.num_envs)
    env_test_ID = ProcgenVecEnvCustom(FLAGS.env_name,
                                      num_levels=200,
                                      mode='easy',
                                      start_level=0,
                                      paint_vel_info=False,
                                      num_envs=FLAGS.num_envs)
    env_test_OOD = ProcgenVecEnvCustom(FLAGS.env_name,
                                       num_levels=0,
                                       mode='easy',
                                       start_level=0,
                                       paint_vel_info=False,
                                       num_envs=FLAGS.num_envs)

    os.environ["WANDB_API_KEY"] = "KEY"
    group_name = "%s_%s" % (FLAGS.env_name, FLAGS.run_id)
    name = "shared_encoder_%s_%s_%d" % (FLAGS.env_name, FLAGS.run_id,
                                        np.random.randint(100000000))

    wandb.init(project='ppo_procgen_jax', entity='entity', config=FLAGS, group=group_name, name=name, mode="disabled" if \
               FLAGS.disable_wandb else "online")
    start_time = time.time()
    algo = PPO(
        num_agent_steps=FLAGS.train_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=FLAGS.seed,
        max_grad_norm=0.5,
        gamma=0.999,
        n_steps=256,
        n_minibatch=8,
        num_envs=FLAGS.num_envs,
        lr=5e-4,
        epoch_ppo=3,
        clip_eps=0.2,
        lambd=0.95,
        entropy_coeff=0.01,
        critic_coeff=1)
        
    state = env.reset()

    for step in tqdm.tqdm(
            range(1, int(FLAGS.train_steps // FLAGS.num_envs + 1))):

        state = algo.step(env, state.astype(jnp.float32) / 255.)
        if algo.is_update():
            loss_actor, loss_critic = algo.update()
            wandb.log(
                {
                    "%s/loss_critic" % (FLAGS.env_name):
                    np.asarray(loss_critic),
                    "%s/loss_actor" % (FLAGS.env_name): np.asarray(loss_actor)
                },
                step=FLAGS.num_envs * step)

        if (step * FLAGS.num_envs) % FLAGS.evaluate_every == 0:
            """
            Evaluate the PPO agent on the test_env
            """
            eval_returns_ID = evaluate(env_test_ID, algo,
                                       FLAGS.num_eval_episodes)
            eval_returns_OOD = evaluate(env_test_OOD, algo,
                                        FLAGS.num_eval_episodes)
            wandb.log({
                "%s/ep_return_200" % (FLAGS.env_name): eval_returns_ID,
                "step": FLAGS.num_envs * step
            })
            wandb.log({
                "%s/ep_return_all" % (FLAGS.env_name): eval_returns_OOD,
                "step": FLAGS.num_envs * step
            })

    wandb.log({"total time": time.time() - start_time})


if __name__ == '__main__':
    app.run(main)
