import numpy as np

class RandomAgent:

    def __init__(self):
        pass

    def step(self, action_history, observations):
        cur_player = int(observations['current_player'])
        actions = observations['legal_actions'][cur_player]
        return np.random.choice(actions)

