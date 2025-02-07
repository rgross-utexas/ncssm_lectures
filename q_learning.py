from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from logger import get_logger
from policy import GreedyPolicy, EGreedyPolicy
from utils import get_datetime_str, generate_video

INFINITY = 10e10
NAME = 'q_learning'

logger = get_logger(NAME)

def q_learning(
        env: gym.Env,
        init_q: np.array,
        alpha: float = 1,
        epsilon: float = .1,
        gamma: float = 1,
        num_episodes: int = 500,
        num_videos: int = 1,
        render_video = False
) -> Tuple[np.array, defaultdict, defaultdict]:
    """
    input:
        env: environment
        init_q: initial Q-values (i.e., Quality of an action given a state)
        alpha: learning rate (i.e., size of the updates), (0, 1])
        epsilon: exploration rate (i.e., the faction of time spent exploring rather than taking the "greedy" action, (0,1]))
        gamma: discount factor (i.e., how important are future rewards? (0, 1]))
        num_episodes: number of episodes to run
        num_videos: number of videos to render, for post-run evaluation
    return:
        q: the learned Q-values
        episode_rewards: an array of rewards by episode, for post-run analysis
        episode_lengths: episode lengths by episode, for post-run analysis
    """

    # make a copy of the initial q
    q = init_q.copy()

    # initialize the policy to an epsilon greedy policy, which will allow
    # for exploration of the environment
    pi = EGreedyPolicy(q, epsilon)

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for episode in tqdm(range(num_episodes)):

        # reset t, the timestep, to 0 and done to false for each episode
        t = 0
        done = False

        # get the initial state
        s_t0 = env.reset()
        s_t0 = s_t0[0]

        # loop until the episode is done, which comes from the environment
        while not done:

            # choose an action based on the current state
            a_t0 = pi.action(s_t0)

            # take a step with the chosen action, getting the next state, the reward, and
            # find out if we are done
            s_t1, r_t1, done, _, _ = env.step(a_t0)

            # update rule based on q: one of the main parts of any RL algorithms!
            q[s_t0, a_t0] = (1 - alpha) * q[s_t0, a_t0] + alpha * (r_t1 + gamma * np.max(q[s_t1]))

            # track rewards and episode lengths
            episode_rewards[episode] += r_t1
            episode_lengths[episode] = t

            # transition to the next state
            s_t0 = s_t1

            # increment t to the next timestep
            t += 1
    
        # if render video is true, generate a video from time to time with the
        # currrent policy
        if render_video & ((episode + 1) % (num_episodes//num_videos) == 0):
            generate_video(env.spec.id, pi.action, episode, NAME)

    return q, episode_rewards, episode_lengths


def render_figure(data: Dict, window: int, label: str, filename: str):

    # render the figure and write it to file
    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for n, data in data.items():

        rewards = pd.Series(data['r'].values()).rolling(window, min_periods=window).mean().to_numpy()
        lengths = pd.Series(data['l'].values()).rolling(window, min_periods=window).mean().to_numpy()

        axes[0].plot(rewards, label=f'{label}={n}')
        axes[1].plot(lengths, label=f'{label}={n}')

    axes[0].legend(loc='lower right')
    axes[0].title.set_text(f"Reward per Episode Over Time ({window} step rolling average)")
    axes[1].legend(loc='upper right')
    axes[1].title.set_text(f"Episode Length Over Time ({window} step rolling average)")

    Path(f'output/images/{NAME}').mkdir(parents=True, exist_ok=True)
    filename = f'output/images/{NAME}/{env.spec.id}_{get_datetime_str()}_{filename}'
    plt.savefig(filename)


def render_q(q):

    lookup = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
    v_spacer = '-------------------------------------------------------------------------'
    h_spacer = '|'
    s : str = h_spacer

    logger.info('Optimal state/action space for the given Q:')
    for cell in range(q.shape[0]):
        col = cell % 12
        s += str(lookup[np.argmax(q[cell])]).rjust(5)
        if col == 11:
            logger.info(v_spacer)
            logger.info(s + h_spacer)
            s = h_spacer
        else:
            s += h_spacer

    logger.info(v_spacer)


if __name__ == '__main__':

    # create a cliff walking env

    env = gym.make('CliffWalking-v0')

    print(f'Cliff Walking: {env}')

    # hyperparameters (i.e., nobs to turn)

    # the learning rate, [0, 1], where 0 means it will not learn and 1 means ignore the past
    alpha = .5

    # the discount factor, [0, 1], where 0 means only consider current rewards and closer to 1
    # means look further out to the future
    gamma = .5

    # how long to train
    num_episodes = 200

    # initial q value to some random values
    init_q = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    # dictionary to stash the data
    data = {}

    # epsilon is the exploration rate, [0, 1], where 0 means no exploration and 1 means
    # completely random actions
    for epsilon in [1e-8, .001, .005, .01, .05, .1, .2]:
        _, rewards, lengths = q_learning(
            env=env,
            alpha=alpha,
            epsilon=epsilon,
            num_episodes=num_episodes,
            init_q=init_q,
            gamma=gamma
        )

        data[epsilon] = {'r': rewards, 'l': lengths}

    # we want to average over a window to smooth things out
    window = 50
    render_figure(data, window, 'epsilon', 'by_epsilon')

    rolling_window = 25

    # create a cliff walking env

    # fix the epsilon and render some videos
    epsilon = .1
    num_videos = 10
    num_episodes = 1000

    env = gym.make('CliffWalking-v0')

    q, _, _ = q_learning(
        env=env,
        alpha=alpha,
        epsilon=epsilon,
        num_episodes=num_episodes,
        num_videos=num_videos,
        init_q=np.zeros(shape=(env.observation_space.n, env.action_space.n)),
        gamma=gamma,
        render_video=True
    )

    # print out the optimal behavior of q
    render_q(q)

    explore = False
    if explore:
        generate_video(env.spec.id, EGreedyPolicy(q, epsilon).action, num_episodes, NAME)
    else:
        generate_video(env.spec.id, GreedyPolicy(q).action, num_episodes, NAME)
