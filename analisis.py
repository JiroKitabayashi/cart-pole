import csv
import gym
import time
from q_learning import digitize_state
import numpy as np

def load_table(res_path):
    with open(res_path) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
    return l

# 環境初期化
q_table = load_table('log/1/learned_Q_table.csv')
env = gym.make('CartPole-v0')
observation = env.reset()
done = False
episode = 0

while not done:
    episode += 1
    env.render()
    time.sleep(0.1)
    state = digitize_state(observation, 6)
    action = np.argmax(q_table[state])
    observation, reward, done, info = env.step(action)

print(episode)
