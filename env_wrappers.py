import gym
import procgen
import cv2, os
import numpy as np
import time
from collections import deque
from typing import Any, Dict, List, Sequence, Tuple
from gym.spaces import Box, Dict, Discrete as DiscreteG

from procgen import ProcgenGym3Env
from procgen import ProcgenEnv
from baselines.common.vec_env import (VecExtractDictObs, VecFrameStack,
                                      VecNormalize, VecEnvWrapper)


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename,
                header={'t_start': self.tstart},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos)

        for i in range(len(dones)):
            if dones[i]:
                # info = infos[i].copy()
                info = {}
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {
                    'r': ret,
                    'l': eplen,
                    't': round(time.time() - self.tstart, 6)
                }
                # for k in self.info_keywords:
                #     epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info
        return obs, rews, dones, newinfos


class WarpFrame(gym.ObservationWrapper):
    def __init__(self,
                 env,
                 width=84,
                 height=84,
                 grayscale=True,
                 dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_colors, self._height, self._width),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(
            original_space.shape) == 3

    def observation(self, obs):
        obs = obs.astype(np.uint8)
        if len(obs.shape) == 4:
            obs = obs[0]
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height),
                           interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        obs = obs.transpose(2, 0, 1)
        return obs


class EpisodeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        env.metadata = {'render.modes': []}
        env.reward_range = (-float('inf'), float('inf'))
        nenvs = env.num_envs
        self.num_envs = nenvs
        super(EpisodeRewardWrapper, self).__init__(env)

        self.aux_rewards = None
        self.num_aux_rews = None

        def reset(**kwargs):
            self.rewards = np.zeros(nenvs)
            self.lengths = np.zeros(nenvs)
            self.aux_rewards = None
            self.long_aux_rewards = None

            return self.env.reset(**kwargs)

        def step(action):
            obs, rew, done, infos = self.env.step(action)

            if self.aux_rewards is None:
                # info = infos[0]
                info = infos
                if 'aux_rew' in info:
                    self.num_aux_rews = len(info['aux_rew'])
                else:
                    self.num_aux_rews = 0

                self.aux_rewards = np.zeros((nenvs, self.num_aux_rews),
                                            dtype=np.float32)
                self.long_aux_rewards = np.zeros((nenvs, self.num_aux_rews),
                                                 dtype=np.float32)

            self.rewards += rew
            self.lengths += 1

            use_aux = self.num_aux_rews > 0

            if use_aux:
                for i, info in enumerate(infos):
                    self.aux_rewards[i, :] += info['aux_rew']
                    self.long_aux_rewards[i, :] += info['aux_rew']

            return obs, rew, done, infos

        self.reset = reset
        self.step = step


class FakeEnv:
    """
    Create a baselines VecEnv environment from a gym3 environment.
    :param env: gym3 environment to adapt
    """
    def __init__(self, env, obs_space, act_space, num_envs):
        self.env = env
        # for some reason these two are returned as none
        # so I passed in a baselinevec object and copied
        # over it's values
        # self.baseline_vec = baselinevec
        self.observation_space = obs_space
        self.action_space = act_space
        self.rewards = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs)
        self.aux_rewards = None
        self.long_aux_rewards = None

    def reset(self):
        _rew, ob, first = self.env.observe()
        if not first.all():
            print("Warning: manual reset ignored")
        return ob

    def step_async(self, ac):
        self.env.act(ac)

    def step_wait(self):
        rew, ob, first = self.env.observe()
        return ob, rew, first, self.env.get_info()

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    @property
    def num_envs(self):
        return self.env.num

    def render(self, mode="human"):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        info = self.env.get_info()
        if mode == "rgb_array" and "rgb" in info:
            return info["rgb"]

    def close(self):
        pass

    # added this in to see if it'll properly call the method for the gym3 object
    def callmethod(self, method: str, *args: Sequence[Any],
                   **kwargs: Sequence[Any]) -> List[Any]:
        return self.env.callmethod(method, *args, **kwargs)


class ProcgenVecEnvCustom():
    def __init__(self,
                 env_name,
                 num_levels,
                 mode,
                 start_level,
                 paint_vel_info=True,
                 num_envs=32):
        self.observation_space_vec = Dict(
            rgb=Box(shape=(3, 64, 64), low=0, high=255))
        self.action_space_vec = DiscreteG(15)

        self.observation_space = Box(shape=(3, 64, 64), low=0, high=255)
        self.action_space = DiscreteG(15)

        self.gym3_env_train = ProcgenEnv(num_envs=num_envs,
                                         env_name=env_name,
                                         num_levels=num_levels,
                                         start_level=start_level,
                                         paint_vel_info=paint_vel_info,
                                         distribution_mode=mode)
                                         
        self.venv_train = VecExtractDictObs(self.gym3_env_train, "rgb")
        self.venv_train = VecMonitor(
            venv=self.venv_train,
            filename=None,
            keep_buf=100,
        )
        self.venv_train = VecNormalize(venv=self.venv_train, ob=False)
        self.env = EpisodeRewardWrapper(self.venv_train)

        self._max_episode_steps = 10_000

    def reset(self):
        o = self.env.reset()
        return o

    def step(self, action):
        o, r, x, info, = self.env.step(action)
        return o, r, x, info