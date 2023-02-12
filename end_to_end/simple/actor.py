import numpy as np
import torch
import torch.nn as nn
import gym

import logging
import sys
import os
from os.path import dirname, abspath
import copy

sys.path.append(dirname(dirname(abspath(__file__))))

from envs.wrappers import StackFrame


class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim=2,
                 feature_dim=32,
                 num_layers_head=2,
                 history_length=1,
                 dropout=0.0):
        """
        state_dim: the state dimension (720 + 2 + 2). we'll drop the action dimension.
        history_length: #timesteps considered in history

        input tensors should be : [batch, history_length, state_dim]
        feature tensors would be : [batch, feature_dim]
        output(action) tensors would be : [batch, action_dim]
        """
        super().__init__()

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=history_length+2,
                                out_channels=feature_dim/8,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/8+2,
                                out_channels=feature_dim/4,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/4+2,
                                out_channels=feature_dim/2,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/2+2,
                                out_channels=feature_dim,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_layers)

        head_layers = []
        flatten_dim = feature_dim * state_dim / 16
        for i in range(num_layers_head):
            head_layers.append(nn.Linear(flatten_dim, flatten_dim, dropout=dropout))
            head_layers.append(nn.ReLU())
        head_layers.append(nn.Linear(flatten_dim, action_dim))
        self.head = nn.Sequential(*head_layers)

    
    def forward(self, states):
        state = state[..., :-2]
        feature = self.conv(state)
        return self.head(feature)


class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim=2,
                 feature_dim=32,
                 num_layers_head=2,
                 history_length=1,
                 dropout=0.0):
        """
        state_dim: the state dimension (720 + 2 + 2)
        history_length: #timesteps considered in history

        input tensors should be : [batch, history_length, state_dim]
        feature tensors would be : [batch, feature_dim]
        output(action) tensors would be : [batch, action_dim]
        """
        super().__init__()

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=history_length+2,
                                out_channels=feature_dim/8,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/8+2,
                                out_channels=feature_dim/4,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/4+2,
                                out_channels=feature_dim/2,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/2+2,
                                out_channels=feature_dim,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_layers)

        head_layers = []
        flatten_dim = feature_dim * state_dim / 16
        for i in range(num_layers_head):
            head_layers.append(nn.Linear(flatten_dim, flatten_dim, dropout=dropout))
            head_layers.append(nn.ReLU())
        head_layers.append(nn.Linear(flatten_dim, action_dim))
        self.head = nn.Sequential(*head_layers)

    
    def forward(self, states):
        state = state[..., :-2]
        feature = self.conv(state)
        return self.head(feature)

def create_env(config):
    env_config = config["env_config"]

    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    
    return env

def select_action(model, obs, noise_var):
    act = model(obs)
    noise = torch.tensor(np.randn(2,) * noise_var)
    return act + noise


def act(
    config,
    actor_index,
    buffers,
    ptr,
    central_actor
):

    def write(path):
        


    try:
        logging.info("Actor %i started.", actor_index)
        training_config = config["training_config"]

        log_path = os.path.join(dirname(dirname(abspath(__file__))), 'data')
        

        model = Actor()
        model.load_state_dict(central_actor.state_dict())

        env = create_env(config)
        step = iter = 0
        episode_count = 0
        obs = env.reset()
        summary = []
        episode_return = 0
        episode_length = 0

        while step < training_config['training_args']['max_step']:
            act = select_action(obs)
            obs_new, rew, done, info = env.step(act)
            obs = obs_new
            episode_return += rew
            episode_length += 1
            step += 1
            if done:
                obs = env.reset()
                summary.append(dict(
                    episode_return = episode_return,
                    episode_length = episode_length,
                    success = info["success"],
                    world = info["world"],
                    collision = info["collision"]
                ))
                episode_length = 0
                episode_return = 0
                episode_count += 1

            if step%training_config['training_args']['update_period'] == 0:
                model.load_state_dict(central_actor.state_dict())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

