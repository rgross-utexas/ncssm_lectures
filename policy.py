from typing import Optional

import numpy as np


class Policy(object):
    def action_prob(self, state: int, action: int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        raise NotImplementedError()

    def action(self, state: int) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError()

    @property
    def p(self) -> np.array:
        raise NotImplementedError()


class RandomPolicy(Policy):
    def __init__(self, num_actions, p=None):
        self._p = p if p is not None else np.array([1 / num_actions] * num_actions)

    def action_prob(self, state: int, action: int) -> float:
        return self.p[action]

    def action(self, state: int) -> int:
        return np.random.choice(len(self._p), p=self._p)

    @property
    def p(self) -> np.array:
        return self._p


class DeterministicPolicy(Policy):

    def __init__(self, p: np.array):
        self._p = p

    def action_prob(self, state: int, action: int) -> float:
        return 1 if self.p[state] == action else 0

    def action(self, state: int) -> int:
        return self.p[state]

    @property
    def p(self) -> np.array:
        return self._p


class GreedyPolicy(Policy):

    def __init__(self, p: np.array):
        self._p = p

    def action_prob(self, state: int, action: int) -> float:
        probs = self._p[state]
        return 1 if np.random.choice(np.flatnonzero(probs == probs.max())) == action else 0

    def action(self, state: int) -> int:
        probs = self._p[state]
        return int(np.random.choice(np.flatnonzero(probs == probs.max())))

    @property
    def p(self) -> np.array:
        return self._p


class EGreedyPolicy(Policy):

    def __init__(self, p: np.array, epsilon: float):

        self._p = p
        self.epsilon = epsilon

    def action_prob(self, state: int, action: int) -> float:
        """
        Assign a probabilty for the given state/action pair.
        """

        num_actions = self._p.shape[1]

        # if random sample is less than epsilon, explore
        if np.random.random_sample() < self.epsilon:

            # assign equal probability to all all actions
            return 1 / num_actions
        else:
            probs = self.p[state]

            # assign probabilty based on whether the action is the most probable 
            if np.random.choice(np.flatnonzero(probs == probs.max())) == action:
                return 1 - self.epsilon + self.epsilon / num_actions
            else:
                return self.epsilon / num_actions

    def action(self, state: int) -> int:
        """
        Choose an action.
        """

        # if random sample is less than epsilon, explore
        if np.random.random_sample() < self.epsilon:

            # choose a random action
            return np.random.choice(self.p.shape[1])
        else:

            if isinstance(state, tuple):
                state = state[0]

            # chooe the most probably action for the given state
            probs = self.p[state]
            return int(np.random.choice(np.flatnonzero(probs == probs.max())))

    @property
    def p(self) -> np.array:
        return self._p


class DoubleEGreedyPolicy(Policy):

    def __init__(self, p1: np.array, p2: np.array, epsilon: float):
        self._p1 = p1
        self._p2 = p2
        self.epsilon = epsilon

    def action_prob(self, state: int, action: int) -> float:

        num_actions = self._p1.shape[1]
        if np.random.random_sample() < self.epsilon:
            return 1 / num_actions
        else:
            if np.argmax(self.p[state]) == action:
                return 1 - self.epsilon + self.epsilon / num_actions
            else:
                return self.epsilon / num_actions

    def action(self, state: int) -> int:
        if np.random.random_sample() < self.epsilon:
            return np.random.choice(self._p1.shape[1])
        else:
            probs = self.p[state]
            return int(np.random.choice(np.flatnonzero(probs == probs.max())))

    @property
    def p(self) -> np.array:
        return (self._p1 + self._p2) / 2
