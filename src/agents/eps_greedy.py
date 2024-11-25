import numpy as np
from agents.agent import BaseAgent


class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, num_actions: int, epsilon: float = 0.1) -> None:
        """
        Initialize the Epsilon-Greedy Agent.

        epsilon: The probability with which the agent will explore (take a random action).
        """
        self._num_actions = num_actions
        self._epsilon = epsilon
        self._q_values = np.zeros(num_actions)
        self._action_counts = np.zeros(num_actions)

    def select_action(self, action_space: np.ndarray, mask: np.ndarray) -> int:
        """
        Select an action based on epsilon-greedy policy.

        action_space: The space of probabilities or values for each action.
        mask: A binary mask indicating valid actions.
        """
        masked_action_space = self.apply_mask(action_space, mask)

        valid_actions = np.where(masked_action_space >= 0)[0]
        if np.random.rand() < self._epsilon:
            action = np.random.choice(valid_actions)
        else:
            action = valid_actions[np.argmax(masked_action_space[valid_actions])]

        return action

    def update(self, action: int, reward: float) -> None:
        self._action_counts[action] += 1
        self._q_values[action] = (
            self._q_values[action]
            + (reward - self._q_values[action]) / self._action_counts[action]
        )
