"""
This is a Work-In-Progress. I believe it runs, but use at your own risk.
"""

from collections import deque
import random
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

from utils import generate_video

NAME = 'deep_q_learning'

# Neural network model for approximating Q-values
class DQN(nn.Module):
    """
    Simple, fully connected NeuralNetwork with a single hidden layer. The input features
    are dependent on the number of states in the space, while the output features are
    dependent on the number of actions in the space. hidden_dims is configurable and defaults
    to 128. If dropout is non-zero, dropout layers are added after each ReLU. negative_slope
    affects the negative slope of the LeakyReLU activations (0 implies regular ReLU).
    """
    def __init__(self, input_dims: int, output_dims, hidden_dims: int = 128,
                dropout: float = 0, negative_slope: float = 0):

        super(DQN, self).__init__()

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
        return self.net(state)

    def get_action_log_prob(self: nn.Module, state: np.ndarray) -> Tuple[int, torch.Tensor]:

        # get the current action probability distribution for the given state
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = F.softmax(self.forward(Variable(state)), dim=1)

        # sample a random action from the state-action probability distribution
        sampled_action = np.random.choice(self.output_dims, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[sampled_action])

        return sampled_action, log_prob


# Function to choose action using epsilon-greedy policy
def select_action(env, policy_net, state, epsilon):

    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        return torch.argmax(q_values).item()  # Exploit


# Function to optimize the model using experience replay
def optimize_model(policy_net, target_net, memory, batch_size, gamma, optimizer):

    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    # Compute Q-values for current states
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def deep_q_learning(
        env: gym.Env,
        learning_rate: float = 0.001,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        gamma: float = .99,
        num_episodes: int = 2500,
        target_update_freq: int = 1000,
        batch_size: int = 64,
        memory_size: int = 10000,
        num_videos: int = 50,
        render_video: bool = True
) -> None:

    # Initialize Q-networks
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # create the two nets
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)

    # set the target new to the same state as the policy net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)

    # main training loop
    step_counter = 0
    for episode in tqdm(range(num_episodes)):

        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:

            # select action and take a step
            action = select_action(env, policy_net, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            
            # store the transition in memory
            memory.append((state, action, reward, next_state, done))
                        
            # optimize model
            optimize_model(policy_net, target_net, memory, batch_size, gamma, optimizer)

            # update target network periodically
            if step_counter % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # update state
            state = next_state
            episode_reward += reward

            step_counter += 1

        # periodically generate a video
        if render_video & ((episode + 1) % (num_episodes//num_videos) == 0):
            generate_video(env.spec.id, policy_net.get_action_log_prob, episode, NAME)

        # decay epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

def main():

    # Create the CartPole environment
    env = gym.make("CartPole-v1")
    env = gym.make("LunarLander-v3")

    deep_q_learning(env)


if __name__ == '__main__':
    main()
