import numpy as np

class RandomAgent:

    def __init__(self):
        pass

    def step(self, legal_actions):
        return np.random.choice(legal_actions)