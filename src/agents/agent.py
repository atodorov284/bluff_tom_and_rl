from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, action_space: np.ndarray, mask: np.ndarray) -> int:
        """
        Abstract method to select an action based on the action space and the mask.
        action_space: A probability distribution or set of probabilities for each action.
        mask: A binary mask of valid actions.
        """
        pass

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the agent's knowledge based on the action taken and the reward received.
        Args:
            action: The arm that was pulled.
            reward: The reward received after pulling the arm.
        """
        pass

    def apply_mask(self, action_space: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply the action mask to the action space by invalidating masked actions."""
        action_space[mask == 0] = -1
        return action_space
