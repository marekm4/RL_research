import gymnasium as gym

from stable_baselines3 import DQN

from dqne import DQNE

for env_name in ["CartPole-v1", "LunarLander-v3"]:
    for timesteps in range(1, 11):
        for algo in [DQN, DQNE]:
            env = gym.make(env_name)
            train = 10
            total_train_reward = 0.0

            for train_step in range(train):

                model = algo("MlpPolicy", env)
                model.learn(total_timesteps=timesteps * 10000)

                episodes = 10
                total_reward = 0.0
                for _ in range(episodes):
                    observation, _ = env.reset()
                    while True:
                        action, _states = model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, _ = env.step(action)

                        total_reward += reward

                        if terminated or truncated:
                            break
                env.close()
                # print("Training Iter:", train_step, "Reward:", total_reward / episodes)
                total_train_reward += total_reward / episodes

            total_train_reward = total_train_reward / train
            print("Env:", env_name, "Algo:", str(algo), "Steps:", timesteps * 10000, "Reward:", total_train_reward)