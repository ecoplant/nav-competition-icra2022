import multiprocessing
from multiprocessing import shared_memory
from os.path import dirname, abspath
import os
import yaml
import logging
import time

import numpy as np
import torch

from buffer import ReplayBuffer
from actor import create_env

def collect(index, ptr, size, config, device):
    buffer = ReplayBuffer(ptr, size, config, device)

    max_steps = 100

    os.environ['ROS_MASTER_URI'] = "http://localhost:"+str(11311+index)
    os.environ['GAZEBO_MASTER_URI'] = "${GAZEBO_MASTER_URI}:-http://localhost:"+str(11345+index)
    print(f"ROS_MASTER_URI={os.environ['ROS_MASTER_URI']} GAZEBO_MASTER_URI={os.environ['GAZEBO_MASTER_URI']}")

    env = create_env(config)
    high = env.action_space.high
    low = env.action_space.low
    bias = (high + low) / 2
    scale = (high - low) / 2

    step = 0
    episode_count = 0
    obs = env.reset()
    print('the shape of observation tensors:' + str(obs.shape))

    while step < max_steps:
        act = np.random.randn((2,)).clip(-1,1)*scale + bias
        obs_new, rew, done, info = env.step(act)
        buffer.add([obs, act, obs_new, rew, done])
        logging.info('Process %i: buffer insertion at position %i',
                    index, buffer.ptr.value)
        obs = obs_new
        step += 1
        if done:
            obs = env.reset()
            episode_count += 1
            logging.info('Episode %i has finished', episode_count)
        
    logging.info("Process %i has successfully terminated", index)

def main():
    logging.basicConfig(format='%(levelname)s %(process)d %(asctime)s %(message)s',level=logging.INFO)

    base_path = dirname(dirname(abspath(__file__)))
    config_path = os.path.join(base_path, 'data/new_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    buffer_size = config['training_config']['buffer_size']
    obs_shape = np.array([config["env_config"]["stack_frame"], config["env_config"]["obs_dim"]])
    act_shape = np.array([2,])

    use_cuda = config["training_config"]["use_cuda"]

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    manager = multiprocessing.Manager()
    buffer_memory = manager.dict(
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

    processes = []
    ctx = multiprocessing.get_context("fork")

    for i in range(2):
        process = ctx.Process(
            target=collect,
            args=(
                i, buffer_memory, ptr, size, config, device
            )
        )
        process.start()
        processes.append(process)
        time.sleep(5)

    time.sleep(30)
    for i in range(buffer.size.value):
        logging.info('%ith state in the buffer: %s',i,str(buffer.buffer_array['state'][i]))

    for i, p in enumerate(processes):
        p.terminate()
        p.join()
        logging.info('process %i has joined',i)

    for k, i in buffer_memory.items():
        i.unlink()
        i.close()
    

if __name__=="__main__":
    main()
