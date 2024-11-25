import numpy as np
from envs.bluff_env import env
from agents.eps_greedy import EpsilonGreedyAgent


def play_bluff_game(num_players=2) -> None:
    game_env = env(num_players=num_players, render_mode="human")

    game_env.reset()
    
    obs, info = game_env.get_initial_observation()
    
    if "action_mask" in info:
        mask = info["action_mask"]
    elif isinstance(obs, dict) and "action_mask" in obs:
        mask = obs["action_mask"]
    else:
        mask = None
    

    print("\n--- Starting the Bluff Game ---\n")
    
    eps_greedy_agent_1 = EpsilonGreedyAgent(num_actions=5, epsilon=0.1)
    
    timestep = 0
    last_action = None

    while not all(game_env.terminations.values()):
        agent = game_env.agent_selection
        
        
        
        action = eps_greedy_agent_1.select_action(np.array([0,1,2,3,4]), mask)
        game_env.step(action)
        obs, reward, termination, truncation, info = game_env.last()
        eps_greedy_agent_1.update(action, reward)
        
        if termination or truncation:
            action = None
        else:
            if "action_mask" in info:
                mask = info["action_mask"]
            elif isinstance(obs, dict) and "action_mask" in obs:
                mask = obs["action_mask"]
            else:
                mask = None
            
    
        print("\n--- Final Results ---")
        print(f"Final Rewards: {game_env.rewards}")
        game_env.close()


if __name__ == "__main__":
    play_bluff_game(num_players=3)
