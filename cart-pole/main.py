import numpy as np
import gymnasium as gym
from collections import namedtuple, deque

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.saving import load_model

class Agent:
    def __init__(self, filepath: str):
        self.model = load_model(filepath)

    def choose_action(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probs, _ = self.model(state)
        return np.random.choice(2, p=np.squeeze(action_probs))

env = gym.make('CartPole-v1', render_mode='human')

agent = Agent('cart_pole.keras')
state, _ = env.reset()
done = False
while not done:
    env.render()
    action = agent.choose_action(state)
    state, _, done, _, _ = env.step(action)