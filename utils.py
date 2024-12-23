from tqdm import tqdm

from gymnasium import Env
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
