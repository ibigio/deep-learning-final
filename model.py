import tensorflow as tf
import pyspiel
from open_spiel.python import rl_environment

class ReinforceAgent(tf.keras.Model):
    def __init__(self):
        super(ReinforceAgent, self).__init__()
        self.action_embedding_size = 50
        self.lstm_embedding_size = 100
        self.max_num_actions = 6 * 5 * 2 # sides of dice * number of dice * number of players

        self.lift_action = tf.keras.layers.Dense(self.action_embedding_size, activation='relu')
        self.lstm = tf.keras.layers.LSTM(self.lstm_embedding_size)
        self.final_activation = tf.keras.layers.Dense(self.max_num_actions, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def step(self, action_history, observations):
        """
        Determines safest call by calling this model.
        """
        return self.call(action_history, observations)

    def call(self, action_history, observations):
        lifted_actions = self.lift_action(action_history)
        lstm_out = self.lstm(inputs=lifted_actions, initial_state=lifted_actions)
        return self.final_activation(tf.concat(lifted_actions, lstm_out))

    def loss(self, probabilities, actions, discounted_rewards):
        gathered = tf.gather_nd(probabilities, [[ind, action] for ind, action in enumerate(actions)])
        neg = -tf.math.log(gathered)
        weighted = neg * discounted_rewards
        return tf.reduce_sum(weighted)