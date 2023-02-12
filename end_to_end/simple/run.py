import argparse
import logging
import os
import time
import timeit
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.sharedctypes import Value
import copy

import torch
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

# create a buffer of torch tensors
def create_buffer(config):
    buffer_size = config["training_config"]["buffer_size"]
    obs_shape = np.array([config["env_config"]["stack_frame"], config["env_config"]["obs_dim"]])
    act_shape = np.array([2,])

    buffers = dict(
        state_mem = shared_memory.SharedMemory(
        create=True, size=np.dtype(np.float32).itemsize*np.prod(obs_shape)*buffer_size, name="state_memory")
    action_mem = shared_memory.SharedMemory(
        create=True, size=np.dtype(np.float32).itemsize*np.prod(act_shape)*buffer_size, name="action_memory")
    next_state_mem = shared_memory.SharedMemory(
        create=True, size=np.dtype(np.float32).itemsize*np.prod(obs_shape)*buffer_size, name="next_state_memory")
    reward_mem = shared_memory.SharedMemory(
        create=True, size=np.dtype(np.float32).itemsize*buffer_size, name="reward_memory")
    done_mem = shared_memory.SharedMemory(
        create=True, size=np.dtype(np.bool).itemsize*buffer_size, name="done_memory")
    )

    ptr = Value('i', 0)

    return buffers, ptr



def train(flags, config):

    use_cuda = config["training_config"]["use_cuda"]

    if use_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        device = torch.device("cpu")

    buffers, ptr = create_buffer(config)

    actor = Actor().to(device)
    actor_delay = copy.deepcopy(actor)
    critic = Critic().to(device)
    critic_delay = copy.deepcopy(critic)
    actor.share_memory()

    actor_processes = []
    ctx = multiprocessing.get_context("fork")

    for i in range(config['training_config']['num_actor']):
        actor_process = ctx.Process(
            target=act,
            args=(
                config,
                i,
                buffers,
                ptr,
                actor
            ),
        )
        actor_process.start()
        actor_processes.append(actor_process)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

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