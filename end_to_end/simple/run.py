import argparse
import logging
import os
from os.path import abspath, dirname
import time
import datetime
import timeit
import multiprocessing
from multiprocessing import shared_memory
import copy
import yaml

import torch
from torch.nn import functional as F
import numpy as np

from actor import *
from buffer import ReplayBuffer

parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--mode", default="train",
                    choices=["train", "test"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--log", action='store_true',
                    help="enable logging")


def train(flags, config):
    #load necessary configs
    use_cuda = config["training_config"]["use_cuda"]

    if use_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        device = torch.device("cpu")

    gamma = config['training_config']['gamma']
    tau = config['training_config']['tau']
    action_low = [config['env_config']['min_v'], config['env_config']['min_w']]
    action_high = [config['env_config']['max_v'], config['env_config']['max_w']]

    action_scale = torch.tensor(
            (action_high - action_low) / 2.0, device=device)
    action_bias = torch.tensor(
        (action_high + action_low) / 2.0, device=device)

    buffer_size = config['training_config']['buffer_size']
    obs_shape = np.array([config["env_config"]["stack_frame"], config["env_config"]["obs_dim"]])
    act_shape = np.array([2,])

    #set paths for logging and checkpoint
    base_path = dirname(dirname(abspath(__file__)))
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    log_path = config['training_config']['log_path']
    log_path = os.path.join(base_path, log_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = os.path.join(log_path, suffix)

    checkpoint_path = config['training_config']['checkpoint_path']
    checkpoint_path = os.path.join(base_path, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    #set a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s %(process)d %(asctime)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler, file_handler)
    logger.info(f'Logging to {log_path}')

    #set a buffer, models and processes
    manager = multiprocessing.Manager()
    buffer_memory = dict(
        state = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*np.prod(obs_shape)*buffer_size, name="state_memory"),
        action = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*np.prod(act_shape)*buffer_size, name="action_memory"),
        next_state = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*np.prod(obs_shape)*buffer_size, name="next_state_memory"),
        reward = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*buffer_size, name="reward_memory"),
        done = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*buffer_size, name="done_memory")
        )
    ptr = manager.Value('i', 0)
    size = manager.Value('i', 0)
    buffer = ReplayBuffer(ptr, size, config, device)

    actor = Actor().to(device)
    actor_target = copy.deepcopy(actor)
    critic = Critic().to(device)
    critic_target = copy.deepcopy(critic)
    actor.share_memory()

    actor_processes = []
    ctx = multiprocessing.get_context("fork")

    for i in range(config['training_config']['num_actor']):
        actor_process = ctx.Process(
            target=act,
            args=(
                config,
                i,
                buffer_memory,
                ptr,
                size,
                actor,
                device
            )
        )
        actor_process.start()
        actor_processes.append(actor_process)

    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=config['training_config']['actor_lr']
    )

    critic_optim = torch.optim.Adam(
        critic.parameters(),
        lr=config['training_config']['critic_lr']
    )

    #start training
    iter, stats = 0, {}
    max_iter = config['training_config']['max_iter']
    actor_optim_interval = \
        config['training_config']['actor_optim_interval']
    log_interval = config['training_config']['log_interval']
    checkpoint_interval = config['training_config']['checkpoint_interval']
    do_validate = config['training_config']['validate']

    while iter < max_iter:
        state, action, next_state, reward, done = buffer.sample()

        with torch.no_grad():
            argmax_action = actor_target(next_state)
            target_q1, target_q2 = critic_target(next_state, argmax_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0-done) * gamma * target_q
        
        action /= action_scale
        action += action_bias
        q1, q2 = critic(state, action)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        if iter % actor_optim_interval == 0:
            actor_loss = -critic.q1(state, actor(state)).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)
                
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)
        
        if iter % log_interval == 0:
            if do_validate:
                validate(actor, config, logger, iter)
            else:
                logger.info('%i iterations')
        
        if iter % checkpoint_interval == 0:
            checkpoint()
        
    
    def checkpoint():
        nonlocal checkpoint_path
        nonlocal actor, critic
        logging.info("Saving checkpoint to %s", checkpoint_path)
        torch.save(
            {
                "actor_state_dict": actor.state_dict,
                "critic_state_dict": critic.state_dict
            },
            checkpoint_path
        )
    
    #join all actor porcesses and close buffer memory
    for p in actor_processes:
        p.terminate()
        p.join()
    
    for _, i in buffer_memory.items():
        i.close()
        i.unlink()

def validate(actor, config, logger, iter, num_episodes=10):
    env = create_env(config)
    actor.eval()

    obs = env.reset()
    episode_return = 0
    episode_length = 0
    returns = []

    logger.info('Start validation at iteration %d', iter)
    while len(returns) < num_episodes:
        action = actor(obs)
        obs_new, rew, done, info = env.step(action)
        episode_return += rew
        episode_length += 1
        if done:
            returns.append(episode_return)
            logger.info(
                "Episode ended after %i steps. Return: %.1f",
                episode_length,
                episode_return,
            )
            episode_return = 0
            episode_length = 0
        else:
            obs = obs_new
    env.close()
    logger.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )

def test(flags, config, num_episodes=10):
    if flags.xpid is None:
        checkpoint_path = "./latest/model.tar"
    else:
        pass

    env = create_env(config)
    model = Actor()
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["actor_state_dict"])

    obs = env.reset()
    episode_return = 0
    episode_length = 0
    returns = []

    while len(returns) < num_episodes:
        action = model(obs)
        obs_new, rew, done, info = env.step(action)
        episode_return += rew
        episode_length += 1
        if done:
            returns.append(episode_return)
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                episode_length,
                episode_return,
            )
            episode_return = 0
            episode_length = 0
        else:
            obs = obs_new
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )


def main(flags):
    config_path = os.path('../data/new_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if flags.mode == "train":
        train(flags,config)
    else:
        test(flags,config)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)