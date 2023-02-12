import argparse
import logging
import os
import time
import timeit
import multiprocessing
from multiprocessing import shared_memory, Lock
from multiprocessing.sharedctypes import Value
import copy

import torch
from torch.nn import functional as F
import numpy as np

from actor import *
from learner import learn

parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--mode", default="train",
                    choices=["train", "test"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

class ReplayBuffer():
    def __init__(self, config, device):
        self.max_size = config['training_config']['buffer_size']
        self.obs_shape = np.array([config["env_config"]["stack_frame"], config["env_config"]["obs_dim"]])
        self.act_shape = np.array([2,])

        self.batch_size = config['training_config']['training_args']['batch_size']

        self.buffers = dict(
        state = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*np.prod(obs_shape)*buffer_size, name="state_memory")
        action = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*np.prod(act_shape)*buffer_size, name="action_memory")
        next_state = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*np.prod(obs_shape)*buffer_size, name="next_state_memory")
        reward = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.float32).itemsize*buffer_size, name="reward_memory")
        done = shared_memory.SharedMemory(
            create=True, size=np.dtype(np.bool).itemsize*buffer_size, name="done_memory")
        )

        self.ptr = Value('i', 0)
        self.size = Value('i',0)

    def sample():
        index = np.random.randint(0, self.size, size = batch_size)
        return (
                torch.FloatTensor(self.buffers['state'][ind]).to(self.device),
                torch.FloatTensor(self.buffers['action'][ind]).to(self.device),
                torch.FloatTensor(self.buffers['next_state'][ind]).to(self.device),
                torch.FloatTensor(self.buffers['reward'][ind]).to(self.device),
                torch.FloatTensor(self.buffers['done'][ind]).to(self.device)
                )


def train(flags, config):
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
            (action_high - action_low) / 2.0, device=self.device)
    action_bias = torch.tensor(
        (action_high + action_low) / 2.0, device=self.device)


    replay_buffer = ReplayBuffer(config, device)

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
                replay_buffer.buffers,
                replay_buffer.ptr,
                replay_buffer.size,
                actor
            ),
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


    step, stats = 0, {}
    max_step = config['training_config']['training_args']['learner_steps']
    actor_optim_period = \
        config['training_config']['training_args']['actor_optim_period']
    log_interval = config['training_config']['log_intervals']

    while step < max_step:
        state, action, next_state, reward, done = replay_buffer.sample()

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

        if step % actor_optim_period == 0:
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
        
        if step % log_interval == 0:
            #todo : make a logging function



    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def test(flags, config, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        agent_outputs = model(observation)
        policy_outputs, _ = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )


def main(flags):
    config_path = os.path('../data/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if flags.mode == "train":
        train(flags,config)
    else:
        test(flags,config)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)