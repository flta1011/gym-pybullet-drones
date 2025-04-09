from gymnasium.envs.registration import register

register(
    id="drone_test_DQN_find_goal-v0",
    entry_point="gym_pybullet_drones.examples.Test_Flo.drone_test_DQN_find_goal:DroneTestDQNFindGoalEnv",
    max_episode_steps=1000,
)
