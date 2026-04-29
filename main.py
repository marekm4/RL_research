import gymnasium as gym

from stable_baselines3 import DQN

from dqne import DQNE

for env_name in ["CartPole-v1"]:
    for timesteps in range(10_000, 40_000, 10_000):
        for algo in [DQN, DQNE]:
            seeds = [42, 137, 256, 512, 999]
            total_train_reward = 0.0

            for seed in seeds:
                env = gym.make(env_name)
                env.reset(seed=seed)
                model = algo("MlpPolicy", env, seed=seed)
                model.learn(total_timesteps=timesteps)

                episodes = 10
                total_reward = 0.0
                for _ in range(episodes):
                    observation, _ = env.reset(seed=seed)
                    while True:
                        action, _states = model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, _ = env.step(action)

                        total_reward += reward

                        if terminated or truncated:
                            break
                env.close()
                total_train_reward += total_reward / episodes

            total_train_reward = total_train_reward / len(seeds)
            print("Env:", env_name, "Algo:", str(algo), "Steps:", timesteps, "Reward:", total_train_reward)