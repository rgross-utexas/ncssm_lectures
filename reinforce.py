"""
TODO: Add better documentation for this file and algorithm!
REINFORCE (or Monte Carlo Policy Gradient) is a model-free, policy-gradient Reinforcement
Learning algorithm.
"""
import argparse
from pathlib import Path
import itertools
from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium import Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, Optimizer, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import settings
from logger import get_logger
from utils import get_datetime_str, generate_video

NAME = 'reinforce'

logger = get_logger(NAME)

# how many times to log the status during a run
NUMBER_OF_LOGS = 100

# output base directory
OUTPUT_BASE_DIR = settings.reinforce.output_base_dir

# see if we can find and use a gpu
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
# this code runs slower on mps, so just use cpu
DEVICE = 'cpu'
logger.info(f'{DEVICE=}')

class PolicyNetwork(nn.Module):
    """
    Simple, fully connected NeuralNetwork with a single hidden layer. The input features
    are dependent on the number of states in the space, while the output features are
    dependent on the number of actions in the space. hidden_dims is configurable and defaults
    to 128. If dropout is non-zero, dropout layers are added after each ReLU. negative_slope
    affects the negative slope of the LeakyReLU activations (0 implies regular ReLU).
    """
    def __init__(self, input_dims: int, output_dims, hidden_dims: int = 128,
                dropout: float = 0, negative_slope: float = 0):

        super(PolicyNetwork, self).__init__()

        self.output_dims = output_dims

        # create the layers, with optional dropouts. LeakyReLU? ¯\_(ツ)_/¯
        layers =  [nn.Linear(input_dims, hidden_dims), nn.LeakyReLU(negative_slope)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        layers.extend([nn.Linear(hidden_dims, hidden_dims), nn.LeakyReLU(negative_slope)])
        if dropout:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims, output_dims))

        # create the simple fully connected net
        self.net = nn.Sequential(*layers)

    def forward(self: nn.Module, state: np.ndarray) -> torch.Tensor:
        return self.net(torch.tensor(state, device=DEVICE, dtype=torch.float32))

    def get_action_log_prob(self: nn.Module, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Helper method specific for RL that samples an action from the neural network based on
        a given state, also returning log-probability for the given action.
        """

        # get the current action probability distribution for the given state
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = F.softmax(self.forward(Variable(state)), dim=1)

        # sample a random action from the state-action probability distribution
        sampled_action = np.random.choice(self.output_dims, p=np.squeeze(probs.detach().cpu().numpy()))
        log_prob = torch.log(probs.squeeze(0)[sampled_action])

        return sampled_action, log_prob


def discount_rewards(
    rewards: List[float],
    gamma: float,
    max_lookahead: int
) -> torch.Tensor:

    # create the powers once outside the loop
    g_powers = np.power(gamma, np.arange(min(len(rewards) - 1, max_lookahead)))

    discounted_rewards = []
    for t in range(len(rewards)):

        g = 0

        # only lookahead at most max_lookahead steps
        t_lookahead = min(t + max_lookahead, len(rewards) - 1)

        # multiply the rewards by the decay and sum to get g
        g  = np.sum(rewards[t:t_lookahead] * g_powers[:t_lookahead - t])

        discounted_rewards.append(g)

    # standardize the discounted rewards, ensuring that we don't have division by 0
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-12)

    return torch.tensor(discounted_rewards, device=DEVICE, dtype=torch.float32)


def update_policy(
    optimizer: Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    rewards: List[float],
    log_probs: List[torch.Tensor],
    gamma: float,
    max_lookahead: int
) -> None:
    """
    Updates the policy net.
    """

    # zero out the gradient
    optimizer.zero_grad()

    # discount the rewards based on gamma, limiting them to max_lookahead steps
    discounted_rewards = discount_rewards(rewards, gamma, max_lookahead)

    # calculate the gradients
    policy_gradients = []
    for log_prob, g in zip(log_probs, discounted_rewards):
        policy_gradients.append(torch.mul(-log_prob, g))

    policy_gradient = torch.stack(policy_gradients).sum()

    # back propogate the policy gradient
    policy_gradient.backward()

    # take a step with the optimizer
    optimizer.step()
    scheduler.step()


def run(**kwargs: Dict) -> None:

    # grab all the paremeters
    env_spec_id = kwargs['env_spec_id']
    inner_dims = kwargs['inner_dims']
    dropout = kwargs['dropout']
    negative_slope = kwargs['negative_slope']
    num_episodes = kwargs['num_episodes']
    lr = kwargs['lr']
    lr_start_factor = kwargs['lr_start_factor']
    lr_end_factor = kwargs['lr_end_factor']
    gamma = kwargs['gamma']
    max_lookahead = kwargs['max_lookahead']
    num_videos = kwargs['num_videos']

    # intialize the gym environment that we are using
    env: Env = gym.make(env_spec_id)

    # initialize tensorboard
    Path(f'{OUTPUT_BASE_DIR}/tb/{NAME}').mkdir(parents=True, exist_ok=True)
    tb_id = f"{OUTPUT_BASE_DIR}/tb/{NAME}/{env.spec.id}_{get_datetime_str()}_{num_episodes}_{inner_dims}_{lr}_{lr_start_factor}_{lr_end_factor}_{gamma}_{max_lookahead}_{dropout}_{negative_slope}_{get_datetime_str()}"
    tb_logger = SummaryWriter(tb_id, flush_secs=5)

    # get number of actions from gym action space
    n_actions = env.action_space.n

    # get the number of states via the length of a state
    state, _ = env.reset()
    n_states = len(state)

    # init the pytorch pieces
    policy_net = PolicyNetwork(n_states, n_actions, inner_dims, dropout, negative_slope).to(DEVICE)
    optimizer = Adam(policy_net.parameters(), lr=lr)

    # add a LR scheduler to allow the LR to get smaller over time to eke out a bit more performance
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=lr_start_factor, end_factor=lr_end_factor, total_iters=num_episodes)

    # keep an absolute step counter for tensorboard metrics
    step_counter = 0
    for episode in tqdm(range(num_episodes), unit='episodes'):

        # init the environment and get it's current state
        state, _ = env.reset()

        # init the data lists
        log_probs = []
        rewards = []

        for step in itertools.count():

            # get an action and action log probability based on the current policy
            action, log_prob = policy_net.get_action_log_prob(state)

            # take a step with the action
            new_state, reward, terminated, truncated, _ = env.step(action)

            # save these values for updating the policy
            log_probs.append(log_prob)
            rewards.append(reward)

            # if we have completed the run (terminated) or gone too many steps (truncated),
            # update the policy
            if terminated or truncated:

                # update the policy based on the rewards and log probabilities of the actions
                update_policy(optimizer, scheduler, rewards, log_probs, gamma, max_lookahead)

                # episode indexed tensorboard metrics
                tb_logger.add_scalar('e_rewards_sum', np.sum(rewards), global_step=episode)
                tb_logger.add_scalar('e_num_steps', step, global_step=episode)
                tb_logger.add_scalar('e_lr', scheduler.get_last_lr()[0], global_step=episode)

                # log information every NUMBER_OF_LOGS episodes
                if (episode + 1) % (num_episodes//NUMBER_OF_LOGS) == 0:
                    logger.info(f"episode: {episode + 1}, reward: {np.round(np.sum(rewards), decimals=3)}")

                # from time to time, generate a video
                if (episode + 1) % (num_episodes//num_videos) == 0:
                    generate_video(env.spec.id, policy_net.get_action_log_prob, episode, NAME, output_base_dir=OUTPUT_BASE_DIR)

                break

            # step counter indexed tensorboard metrics
            tb_logger.add_scalar('log_prob', log_prob, global_step=step_counter)
            tb_logger.add_scalar('prob', torch.exp(log_prob), global_step=step_counter)
            tb_logger.add_scalar('reward', reward, global_step=step_counter)

            step_counter += 1

            # update the state
            state = new_state


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(prog='reinforce')

    # currently there are no args to parse
    args = parser.parse_args()

    return args


def main():

    logger.info('Starting...')

    # this just a place holder for if/when we need cli args
    _ = parse_args()

    # run each given environment
    for env in settings.reinforce.envs:
        env_dict = dict(env)
        logger.info(f'Running {env_dict}...')
        run(**env_dict)

    logger.info('Done.')


if __name__ == '__main__':
    main()
