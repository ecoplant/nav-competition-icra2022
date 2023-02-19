import multiprocessing
from multiprocessing import shared_memory

import numpy as np
import torch

class ReplayBuffer():
    def __init__(self,
                buffer_momory,
                ptr, size,
                config, device):
        self.device = device
        self.batch_size = config['training_config']['batch_size']
        self.buffer_size = config['training_config']['buffer_size']
        
        self.buffer_memory = buffer_momory

        self.ptr = ptr
        self.size = size

    def sample(self):
        index = np.random.randint(0, self.size, size = self.batch_size)
        return (
                torch.FloatTensor(self.buffers['state'][index]).to(self.device),
                torch.FloatTensor(self.buffers['action'][index]).to(self.device),
                torch.FloatTensor(self.buffers['next_state'][index]).to(self.device),
                torch.FloatTensor(self.buffers['reward'][index]).to(self.device),
                torch.FloatTensor(self.buffers['done'][index]).to(self.device)
                )
    
    def add(self, data):
        self.buffer_memory['state'][self.ptr] = data[0]
        self.buffer_memory['action'][self.ptr] = data[1]
        self.buffer_memory['next_state'][self.ptr] = data[2]
        self.buffer_memory['reward'][self.ptr] = data[3]
        self.buffer_memory['done'][self.ptr] = data[4]

        self.prt.value = (self.ptr.value +1)%self.buffer_size
        self.size.value = min(self.size.vlaue, self.buffer_size)
