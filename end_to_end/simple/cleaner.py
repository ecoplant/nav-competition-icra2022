from multiprocessing import shared_memory

# in cases that some SharedMemory instance is not unlink, unlink the memory.
buffer_memory = dict(
        state = shared_memory.SharedMemory(name="state_memory"),
        action = shared_memory.SharedMemory(name="action_memory"),
        next_state = shared_memory.SharedMemory(name="next_state_memory"),
        reward = shared_memory.SharedMemory(name="reward_memory"),
        done = shared_memory.SharedMemory(name="done_memory")
        )

for _, item in buffer_memory.item():
    item.unlink()