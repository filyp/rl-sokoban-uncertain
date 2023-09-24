# %% [markdown]
# TODO:
# - calculate the actual num of boxes
# - one-hot encoding
#
# not so important TODO:
# - registed my new env instead of modifying boxoban
# - use some longer bursts of exploration
# - the longer you are in the episode, the higher exploration rate should be
#
# done:
# - some clever adaptive curriculum
# TODO how is it possible to have O states?? is it true or display bug?

# %%
import argparse
import sys
import os
from pathlib import Path
import time
import datetime
import torch_ac
import numpy as np
import tensorboardX

import subprocess

repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
repo_root = Path(repo_root)
sys.path.append(repo_root.as_posix())
sys.path.append((repo_root / "gym-sokoban").as_posix())

import utils
from utils import device
from model import ACModel

import gymnasium as gym
import gym_sokoban
from gym_sokoban.envs.sokoban_uncertain import MapSelector

os.environ["RL_STORAGE"] = (repo_root / "storage").as_posix()


# %%
class Args:
    algo = "ppo"  # algorithm to use: a2c | ppo (REQUIRED)
    # https://github.com/mpSchrader/gym-sokoban/blob/default/docs/variations/Boxoban.md
    env = "SokobanUncertain"  # name of the environment to train on (REQUIRED)
    # env = 'MiniGrid-DoorKey-5x5-v0'    # nrme of the environment to train on (REQUIRED)
    maps = repo_root / "custom_maps/1player_2color_5x5"

    model = "a"  # name of the model (default: {ENV}_{ALGO}_{TIME})
    seed = 1  # random seed (default: 1)
    log_interval = 1  # number of updates between two logs (default: 1)
    save_interval = 10  # number of updates between two saves (default: 10, 0 means no saving)
    procs = 16  # number of processes (default: 16)
    frames = 1600000  # number of frames of training (default: 1e7)
    max_episode_steps = 10  # maximum number of steps per episode (default: 200)

    # parameters for main algorithm
    epochs = 4  # number of epochs for PPO (default: 4)
    batch_size = 256  # batch size for PPO (default: 256)
    frames_per_proc = 128  # number of frames per process before update (default: 5 for A2C and 128 for PPO)
    discount = 0.99  # discount factor (default: 0.99)
    lr = 0.001  # learning rate (default: 0.001)
    gae_lambda = 0.95  # lambda coefficient in GAE formula (default: 0.95, 1 means no gae)
    entropy_coef = 0.01  # entropy term coefficient (default: 0.01)
    value_loss_coef = 0.5  # value loss term coefficient (default: 0.5)
    max_grad_norm = 0.5  # maximum norm of gradient (default: 0.5)
    optim_eps = 1e-8  # Adam and RMSprop optimizer epsilon (default: 1e-8)
    optim_alpha = 0.99  # RMSprop optimizer alpha (default: 0.99)
    clip_eps = 0.2  # clipping epsilon for PPO (default: 0.2)
    recurrence = 1  # number of time-steps gradient is backpropagated (default: 1).
    # If > 1, a LSTM is added to the model to have memory.
    text = False  # add a GRU to the model to handle text input


args = Args()

args.mem = args.recurrence > 1

# Set run dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"
model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set device
txt_logger.info(f"Device: {device}\n")

# Load environments
map_selector = MapSelector(
    custom_maps=args.maps,
    # curriculum_cutoff=48*40,
    # hardcode_level=-1,  # None
)
envs = []
for i in range(args.procs):
    env = gym.make(
        args.env,
        max_episode_steps=args.max_episode_steps,
        map_selector=map_selector,
    )
    # TODO seeding may be broken, because numpy generator is global?
    env.reset(seed=args.seed + i)
    envs.append(env)
txt_logger.info("Environments loaded\n")

# Load training status
try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# %%
# Load model

acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

# Load algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(
        envs,
        acmodel,
        device,
        args.frames_per_proc,
        args.discount,
        args.lr,
        args.gae_lambda,
        args.entropy_coef,
        args.value_loss_coef,
        args.max_grad_norm,
        args.recurrence,
        args.optim_alpha,
        args.optim_eps,
        preprocess_obss,
    )
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(
        envs,
        acmodel,
        device,
        args.frames_per_proc,
        args.discount,
        args.lr,
        args.gae_lambda,
        args.entropy_coef,
        args.value_loss_coef,
        args.max_grad_norm,
        args.recurrence,
        args.optim_eps,
        args.clip_eps,
        args.epochs,
        args.batch_size,
        preprocess_obss,
    )
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters
    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs
    if update % args.log_interval == 0:
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["curr_cutoff"]
        data += [len(map_selector.curriculum_scores)]
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        # print which levels are learned well
        to_print = ""
        for score in map_selector.curriculum_scores:
            if score < 10:
                to_print += str(score)
            else:
                to_print += " "
        to_print += ">"
        # max width is 48, wrap around
        print()
        for i in range(0, len(to_print), 48):
            print(f"{i//48:3} |{to_print[i:i+48]}|")

        # txt_logger.info(
        #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:6.2f} {:6.2f} {:6.2f} {:6.2f} | F:μσmM {:5.2f} {:5.2f} {:5.2f} {:5.2f} | curr_cutoff {:2.0f} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
        #     .format(*data))

        # only show num frames, fps, mean F
        txt_logger.info(
            f"F{num_frames:07} | FPS {fps:4.0f} | mean frames per episode: {num_frames_per_episode['mean']:4.1f}"
        )

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status
    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {
            "num_frames": num_frames,
            "update": update,
            "model_state": acmodel.state_dict(),
            "optimizer_state": algo.optimizer.state_dict(),
        }
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        # txt_logger.info("Status saved")

    if num_frames_per_episode["mean"] < 3.5:
        map_selector.grow_curriculum(6)
