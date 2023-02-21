import gym
import numpy as np
from tqdm import tqdm

from wann import WAN
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2")

wan = WAN(-1.5, env.observation_space.sample().shape[0], env.action_space.n)

training_steps = 500
n_steps_per_mutations = 25
cutoff = 0.37 * training_steps

test_episodes = 50

best_reward = float('-inf')
epoch = 0
means = []
rewards = []

for epoch in tqdm(range(training_steps), desc='Training'):
    wan.tune_weights()
    mutation_rewards = []
    for _epoch in range(n_steps_per_mutations):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = int(wan.get_action(state)) % env.action_space.n
            state, reward, done, info = env.step(action)
            total_reward += reward

        mutation_rewards.append(total_reward)

    mean = np.mean(mutation_rewards)
    means.append(mean)

    if mean > best_reward:
        best_reward = np.mean(mutation_rewards)
        if epoch >= cutoff:
            break

print(f'Best reward: {best_reward} found at epoch {epoch}')
plt.plot(means)
plt.show()
input('Press enter to continue...')

for epoch in range(test_episodes):
    state = env.reset()
    done = False
    running_reward = 0
    while not done:
        action_now = int(wan.get_action(state)) % env.action_space.n
        _state, reward, done, _ = env.step(action_now)
        state = _state
        running_reward += reward
        env.render()
    rewards.append(running_reward)

plt.plot(rewards)
plt.show()
