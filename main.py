import gymnasium as gym

from stable_baselines3 import DQN

from dqne import DQNE

for algo in [DQNE, DQN]:
    env = gym.make("CartPole-v1")
    train = 100
    total_train_reward = 0.0

    for train_step in range(train):

        model = algo("MlpPolicy", env)
        model.learn(total_timesteps=10000)

        episodes = 100
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
        print("Training Iter:", train_step, "Reward:", total_reward / episodes)
        total_train_reward += total_reward / episodes

    total_train_reward = total_train_reward / train
    print("Total Training Reward: ", total_train_reward)