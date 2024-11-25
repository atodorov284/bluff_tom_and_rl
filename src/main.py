from envs.bluff_env import env


def play_bluff_game(num_players=2) -> None:
    game_env = env(num_players=num_players, render_mode="human")

    game_env.reset()

    print("\n--- Starting the Bluff Game ---\n")

    while not all(game_env.terminations.values()):
        agent = game_env.agent_selection
        action = game_env.action_space(agent).sample()
        game_env.step(action)

    print("\n--- Final Results ---")
    print(f"Final Rewards: {game_env.rewards}")
    game_env.close()


play_bluff_game(num_players=2)
