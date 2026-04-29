import sys

import gymnasium as gym

from stable_baselines3 import DQN
from dqne import DQNE

import matplotlib.pyplot as plt

results = {}

for env_name in ["CartPole-v1", "LunarLander-v3"]:
    for timesteps in range(10_000, 60_000, 10_000):
        for algo in ["DQN", "DQNE"]:
            seeds = [42, 137, 256, 512, 999]
            seeds = [42]
            total_train_reward = 0.0

            for seed in seeds:
                env = gym.make(env_name)
                env.reset(seed=seed)
                model = getattr(sys.modules[__name__], algo)("MlpPolicy", env, seed=seed)
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
            if env_name not in results:
                results[env_name] = {}
            if algo not in results[env_name]:
                results[env_name][algo] = {}
            if "s" not in results[env_name][algo]:
                results[env_name][algo]["s"] = []
            results[env_name][algo]["s"].append(timesteps)
            if "r" not in results[env_name][algo]:
                results[env_name][algo]["r"] = []
            results[env_name][algo]["r"].append(total_train_reward)
            print("Env:", env_name, "Algo:", str(algo), "Steps:", timesteps, "Reward:", total_train_reward)

print(results)

for env_name, env_results in results.items():
    fig, ax = plt.subplots()
    for algo_name, algo_results in env_results.items():
        ax.plot(algo_results['s'], algo_results['r'], label=algo_name)

    ax.set(xlabel='Steps', ylabel='Reward', title=env_name)
    ax.grid()
    ax.legend()

    fig.savefig(f"results-{env_name}.png")
    plt.show()