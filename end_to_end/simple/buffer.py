import multiprocessing
from multiprocessing import shared_memory

import numpy as np
import torch

class ReplayBuffer():
    def __init__(self,
                ptr, size,
                config, device):
        self.device = device
        self.batch_size = config['training_config']['batch_size']
        self.buffer_size = config['training_config']['buffer_size']
        self.obs_shape = np.array([config["env_config"]["stack_frame"], config["env_config"]["obs_dim"]])
        self.act_shape = np.array([2,])
        
        self.buffer_memory = dict(
        state = shared_memory.SharedMemory(name="state_memory"),
        action = shared_memory.SharedMemory(name="action_memory"),
        next_state = shared_memory.SharedMemory(name="next_state_memory"),
        reward = shared_memory.SharedMemory(name="reward_memory"),
        done = shared_memory.SharedMemory(name="done_memory")
        )

        self.buffer_array = dict(
        state = np.ndarray((self.buffer_size,)+self.obs_shape.shape, dtype=np.float32,
        buffer=self.buffer_memory['state'].buf),
        action = np.ndarray((self.buffer_size,)+self.act_shape.shape, dtype=np.float32,
        buffer=self.buffer_memory['action'].buf),
        next_state = np.ndarray((self.buffer_size,)+self.obs_shape.shape, dtype=np.float32,
        buffer=self.buffer_memory['next_state'].buf),
        reward = np.ndarray((self.buffer_size,), dtype=np.float32,
        buffer=self.buffer_memory['reward'].buf),
        done = np.ndarray((self.buffer_size,), dtype=np.float32,
        buffer=self.buffer_memory['done'].buf)
        )

        self.ptr = ptr
        self.size = size

    def sample(self):
        index = np.random.randint(0, self.size, size = self.batch_size)
        return (
                torch.FloatTensor(self.buffer_array['state'][index]).to(self.device),
                torch.FloatTensor(self.buffer_array['action'][index]).to(self.device),
                torch.FloatTensor(self.buffer_array['next_state'][index]).to(self.device),
                torch.FloatTensor(self.buffer_array['reward'][index]).to(self.device),
                torch.FloatTensor(self.buffer_array['done'][index]).to(self.device)
                )
    
    def add(self, data):
        self.buffer_array['state'][self.ptr] = data[0]
        self.buffer_array['action'][self.ptr] = data[1]
        self.buffer_array['next_state'][self.ptr] = data[2]
        self.buffer_memory['reward'][self.ptr] = data[3]
        self.buffer_array['done'][self.ptr] = data[4]

        self.prt.value = (self.ptr.value +1)%self.buffer_size
        self.size.value = min(self.size.vlaue, self.buffer_size)
