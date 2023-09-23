# %%
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from pynput import keyboard

import subprocess
repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
repo_root = Path(repo_root)
sys.path.append(repo_root.as_posix())
sys.path.append((repo_root / "gym-sokoban").as_posix())

import utils
from utils import device

import gym_sokoban
from gym_sokoban.envs.sokoban_uncertain import MapSelector

os.environ["RL_STORAGE"] = (repo_root / "storage").as_posix()

os.environ["QT_QPA_PLATFORM"] = "xcb" # to prevent warnings on wayland

# %%
# Parse arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--env", help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False, help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1, help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None, help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000, help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False, help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False, help="add a GRU to the model")
# fmt: on

_args = argparse.Namespace(
    env="SokobanUncertain",
    model="typek",
    episodes=10,
    manual=False,
    gif="test",
    maps = repo_root / "custom_maps/1player_2color_5x5",
    max_episode_steps = 10,  # maximum number of steps per episode (default: 200)
    seed=2,
)
args = parser.parse_args([], namespace=_args)

# Set device
print(f"Device: {device}\n")

map_selector = MapSelector(
    custom_maps=args.maps, 
    curriculum_cutoff=20,
    # hardcode_level=-1,  # None
)
# Load environment
env = gym.make(
    args.env, 
    max_episode_steps=args.max_episode_steps,
    map_selector=map_selector,
    # seed=args.seed,
    # render_mode="human"
)
env.reset(seed=args.seed)
print("Environment loaded\n")

# %%

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(
    env.observation_space,
    env.action_space,
    model_dir,
    argmax=args.argmax,
    use_memory=args.memory,
    use_text=args.text,
)
print("Agent loaded\n")


# %%
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Run the agent

def get_manual_action(obs):
    # The event listener will be running in this block
    with keyboard.Events() as events:
        for event in events:
            # ignore release events
            if type(event) == keyboard.Events.Release:
                continue

            if event.key == keyboard.Key.up:
                # print("up")
                return 0
            elif event.key == keyboard.Key.down:
                # print("down")
                return 1
            elif event.key == keyboard.Key.left:
                # print("left")
                return 2
            elif event.key == keyboard.Key.right:
                # print("right")
                return 3
            elif event.key == keyboard.Key.esc:
                # print("esc")
                return -1
            else:
                # print('Received event {}'.format(event))
                pass


if args.gif:
    from array2gif import write_gif
    frames = []

# Create a window to view the environment
fig, ax = plt.subplots()
plt.ion()
plt.show(block=False)


def update_fig(img):
    ax.imshow(img)
    plt.draw()
    plt.pause(0.03)


for episode in range(args.episodes):
    obs, _ = env.reset()

    while True:
        # process the image
        img = env.render()
        update_fig(img)
        if args.gif:
            frames.append(np.moveaxis(img, 2, 0))

        # act on the observation
        if args.manual:
            action = get_manual_action(_)
            if action == -1:
                break  # exit on esc
        else:
            action = agent.get_action(obs)
        
        # get state after action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        # # print reward
        # print(f"Reward: {env.unwrapped.reward_last:.2f}")

        if done:
            if args.manual:
                # wait for keypress to close the window
                _ = get_manual_action(_)
            break

    # process the image
    img = env.render()
    update_fig(img)
    if args.gif:
        frames.append(np.moveaxis(img, 2, 0))
        frames.extend([frames[-1]] * 5)

if args.gif:
    print("Saving gif... ", end="")
    frames = np.array(frames)
    # scale up
    scale = 1 # 40
    frames = np.repeat(frames, scale, axis=2)
    frames = np.repeat(frames, scale, axis=3)
    write_gif(frames, args.gif + ".gif", fps=1 / args.pause)
    print("Done.")
    print(np.array(frames).shape)

plt.ioff()
plt.close()


