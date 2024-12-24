from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
from gymnasium import Env
import numpy as np
import torch
from tqdm import tqdm

from policy import Policy


def generate_trajectories(env: Env, pi: Policy, num_trajectories: int):

    trajs = []
    for _ in tqdm(range(num_trajectories)):
        states, actions, rewards, done = [env.reset()[0]], [], [], []

        while not done:
            a = pi.action(states[-1])
            s, r, done, _, _ = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

        traj = list(zip(states[:-1], actions, rewards, states[1:]))
        trajs.append(traj)

    return trajs


def get_datetime_str() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def show_mps() -> None:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")


def generate_video(env_spec_id: str, policy_fn, episode: int, name: str) -> None:
    """
    Generate an mp4 video for the given policy, and write it to file. This is a fun way
    to see how the algoirthm is doing.
    """

    # render episodes based on the trained policy
    env = gym.make(env_spec_id, render_mode='rgb_array')
    env.metadata['render_fps'] = 120

    # record the frames so we can create a video
    frames = []
    total_reward = 0

    # initialize/reset the environment and get it's state
    state, _ = env.reset()
    while True:

        action = policy_fn(state)
        if isinstance(action, tuple):
            action = action[0]

        new_state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward

        if terminated or truncated:
            break

        state = new_state

    np_frames = np.array(frames)
    Path(f'output/images/{name}').mkdir(parents=True, exist_ok=True)
    filename = f'output/images/{name}/{env.spec.id}_{get_datetime_str()}_{episode + 1}_episodes_{int(total_reward)}_reward.mp4'

    fps = 30
    height = np_frames.shape[2]
    width = np_frames.shape[1]

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
    for i in range(np_frames.shape[0]):
        data = np_frames[i, :, :, :]
        out.write(data)

    out.release()
