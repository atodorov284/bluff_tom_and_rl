from envs.bluff_env import env



def play_bluff_game(num_players=2) -> None:
    game_env = env(num_players=num_players, render_mode="human")

    game_env.reset()

    print("\n--- Starting the Bluff Game ---\n")
    

    while not all(game_env.terminations.values()):
        agent = game_env.agent_selection
        
        obs, reward, termination, truncation, info = game_env.last()
        if termination or truncation:
            action = None
            
        else:
            if "action_mask" in info:
                mask = info["action_mask"]
            elif isinstance(obs, dict) and "action_mask" in obs:
                mask = obs["action_mask"]
            else:
                mask = None
            
            action = game_env.action_space(agent).sample(mask)
        game_env.step(action)

    print("\n--- Final Results ---")
    print(f"Final Rewards: {game_env.rewards}")
    game_env.close()


if __name__ == "__main__":
    play_bluff_game(num_players=2)
