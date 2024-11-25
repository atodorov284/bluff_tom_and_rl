import numpy as np
from envs.bluff_env import env
from agents.eps_greedy import EpsilonGreedyAgent


eps_greedy_agent_1 = EpsilonGreedyAgent(num_actions=5, epsilon=0.5)
eps_greedy_agent_2 = EpsilonGreedyAgent(num_actions=5, epsilon=0.1)


def play_bluff_game(num_players=2, episodes=100) -> None:
    game_env = env(num_players=num_players)
    count_player_0, count_player_1 = 0, 0
    for episode in range(episodes):
        game_env.reset()

        obs, info = game_env.get_initial_observation()

        if "action_mask" in info:
            mask = info["action_mask"]
        elif isinstance(obs, dict) and "action_mask" in obs:
            mask = obs["action_mask"]
        else:
            mask = None

        timestep = 0

        while True:
            agent = game_env.agent_selection
            
            if "action_mask" in info:
                mask = info["action_mask"]
            elif isinstance(obs, dict) and "action_mask" in obs:
                mask = obs["action_mask"]
            else:
                mask = None

            if timestep % 2 == 1:
                action = eps_greedy_agent_1.select_action(np.array([0, 1, 2, 3, 4]), mask)
            else:
                #action = game_env.action_space(agent).sample(mask)
                action = eps_greedy_agent_2.select_action(np.array([0, 1, 2, 3, 4]), mask)
                
            
                
            game_env.step(action)
            obs, reward, termination, truncation, info = game_env.last()        
                
            if timestep % 2 == 1:
                eps_greedy_agent_1.update(action, reward)
                
            else: 
                eps_greedy_agent_2.update(action, reward)

            if termination or truncation:
                if game_env.agent_selection == 'player_0':
                    count_player_0 += 1
                if game_env.agent_selection == 'player_1':
                    count_player_1 += 1
                break
            
            timestep += 1
        
    print(count_player_0, count_player_1)
        
        


if __name__ == "__main__":
    for i in range(1):
        play_bluff_game(num_players=2, episodes=1000)

    print(eps_greedy_agent_1._q_values)
    print(eps_greedy_agent_2._q_values)
