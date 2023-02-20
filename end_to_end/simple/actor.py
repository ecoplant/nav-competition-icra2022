import numpy as np
import torch
import torch.nn as nn
import gym

import logging
import traceback
import sys
import os
from os.path import dirname, abspath
import copy

sys.path.append(dirname(dirname(abspath(__file__))))

from envs.wrappers import StackFrame
from buffer import ReplayBuffer


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
        conv_layers.append(nn.Conv1d(in_channels=history_length,
                                out_channels=feature_dim/8,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/8,
                                out_channels=feature_dim/4,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/4,
                                out_channels=feature_dim/2,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(in_channels=feature_dim/2,
                                out_channels=feature_dim,
                                kernel_size=3, stride=2, padding=1))
        conv_layers.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_layers)

        head_layers = []
        dummy = torch.zeros([1,history_length,state_dim-4])
        dummy = self.conv(dummy)
        flatten_dim = torch.prod(dummy) + 2 * history_length
        for i in range(num_layers_head):
            head_layers.append(nn.Linear(flatten_dim, flatten_dim, dropout=dropout))
            head_layers.append(nn.ReLU())
        head_layers.append(nn.Linear(flatten_dim, action_dim))
        head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)
     
    
    def forward(self, state):
        laser = state[..., :-4]
        pos = state[..., -4:-2]
        feature = self.conv(laser)
        feature = torch.reshape(feature, [feature.shape[0],-1])
        pos = torch.reshape(pos, [pos.shape[0],-1])
        return self.head(torch.cat([feature, pos], dim=1))


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
        conv_layers.append(nn.ReLU)
        self.conv = nn.Sequential(*conv_layers)

        head_layers = []
        dummy = torch.zeros([1,history_length,state_dim-4])
        dummy = self.conv(dummy)
        flatten_dim = torch.prod(dummy) + 4 * history_length
        
        for i in range(num_layers_head):
            head_layers.append(nn.Linear(flatten_dim, flatten_dim, dropout=dropout))
            head_layers.append(nn.ReLU())
        head_layers.append(nn.Linear(flatten_dim, 1))
        self.head = nn.Sequential(*head_layers)

    
    def forward(self, state):
        laser = state[..., :-4]
        others = state[..., -4:]
        feature = self.conv(laser)
        feature = torch.reshape(feature, [feature.shape[0],-1])
        others = torch.reshape(others, [others.shape[0],-1])
        return self.head(torch.cat([feature, others], dim=1))

def create_env(config):
    env_config = config["env_config"]

    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    
    return env

def select_action(model, obs, noise):
    act = model(obs)
    noise = torch.tensor(np.randn(2,) * noise)
    return torch.clip(act + noise, min=-1, max=1).cpu().numpy()


def act(
    config,
    actor_index,
    ptr,
    size,
    central_actor,
    device
):

    try:
        logging.basicConfig(level='info', format='%(levelname)s %(process)d %(asctime)s %(message)s')
        logging.info("Actor %i started.", actor_index)

        max_steps = config['training_config']['actor_max_steps']
        update_interval = config['training_config']['actor_update_interval']
        noise_start = config['training_config']['exploration_noise_start']
        noise_end = config['training_config']['end']

        model = Actor()
        model.load_state_dict(central_actor.state_dict())

        buffer = ReplayBuffer(ptr, size, config, device)

        env = create_env(config)
        high = env.action_space.high
        low = env.action_space.low
        bias = (high + low) / 2
        scale = (high - low) / 2

        step = 0
        episode_count = 0
        obs = env.reset()
        # summary = []
        episode_return = 0
        episode_length = 0

        while step < max_steps:
            noise = noise_start + (noise_end-noise_start)/max_steps*step
            act = select_action(model, obs, noise)
            act = act*scale + bias
            obs_new, rew, done, info = env.step(act)
            buffer.add([obs, act, obs_new, rew, done])
            obs = obs_new
            episode_return += rew
            episode_length += 1
            step += 1
            if done:
                obs = env.reset()
                # summary.append(dict(
                #     episode_return = episode_return,
                #     episode_length = episode_length,
                #     success = info["success"],
                #     world = info["world"],
                #     collision = info["collision"]
                # ))
                episode_length = 0
                episode_return = 0
                episode_count += 1

            if step%update_interval == 0:
                model.load_state_dict(central_actor.state_dict())
        
        for _, i in buffer.buffer_memory.item():
            i.close()
        logging.info("Actor %i has successfully terminated", actor_index)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

