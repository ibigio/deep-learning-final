import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        Comments.
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        
        # TODO: Define network parameters and optimizer
        pass
        
    @tf.function
    def call(self, states, hand):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [history_length, state_size] dimensioned array representing the history
            of states of an episode. Each state is composed as follows:
            - state_size      63  (2 players + 61 action size)
            - state[0:2]      one-hot of player who made the call ([0,1] or [1,0])
            - state[2:63]     one-hot of action

        :param hand: A 1D one-hot representation of the players hand (len 252, one per unique hand).

        :return: A [episode_length,num_actions] matrix representing the probability distribution over
            actions of each state in the episode
        """
        # TODO: implement call
        # 1. Raise each state in states with an embedding layer.
        # 2. Raise the hand with a different embedding layer.
        # 3. Feed states through an LSTM, and get the last hidden state.
        # 4. Feed hand through a Dense layer.
        # 5. Concat output of LSTM and output of Dense layer.
        # 6. Feed concat output through one (or more) Dense layer(s).
        # 7. Feed output through Dense layer to size num_actions (61), and call softmax.
        # 8 (maybe here, or maybe in train/test). Mask out illegal calls, and re-normalize.
        # 9. Return matrix of probability distribution over actions.
        pass

    def loss(self, states, actions, discounted_rewards):
        """
        Comments.
        """
        # TODO: implement loss


