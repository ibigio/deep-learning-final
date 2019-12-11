import os
import gym
import numpy as np
import tensorflow as tf
from tf.keras.layers import Embedding, GRU, Dense

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions
        self.event_size = 63
        self.hand_size = 252

        self.event_embedding_size = 100
        self.hand_embedding_size = 100

        self.event_hidden_size = 100
        self.hand_hidden_size = 100

        self.concat_hidden_size = 100
        
        # TODO: Define network parameters and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.event_embedding = Dense(self.event_embedding_size, activation='relu')
        self.hand_embedding = Dense(self.hand_embedding_size, activation='relu')

        self.actor_event_gru = GRU(self.event_hidden_size, return_sequences=True, return_state=True)
        self.actor_hand_dense = Dense(self.hand_hidden_size, activation='relu')
        self.actor_concat_dense = Dense(self.concat_hidden_size, activation='relu')
        self.actor_output_dense = Dense(self.num_actions, activation='softmax')

        self.critic_event_gru = GRU(self.event_hidden_size, return_sequences=True, return_state=True)
        self.critic_hand_dense = Dense(self.hand_hidden_size, activation='relu')
        self.critic_concat_dense = Dense(self.concat_hidden_size, activation='relu')
        self.critic_output_dense = Dense(1)

    @tf.function
    def call(self, event_history, player_info_state):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        of each state in the episode
        """
        # Embed hand and events.
        # event_embedding = self.event_embedding(event_history)
        hand_embedding = self.hand_embedding(player_info_state)

        # Pass events though gru, and hand through dense layer.
        gru_out, gru_state = self.actor_event_gru(inputs=event_history)
        dense_hand = self.actor_hand_dense(hand_embedding)

        # Concatonate and pass through dense layer.
        concatted = tf.concat(gru_out, dense_hand)
        dense_concatted = self.actor_concat_dense(concatted)

        # Return final probabilities
        output = self.actor_output_dense(dense_concatted)
        return output

    def value_function(self, event_history, player_info_state):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode
        :return: A [episode_length] matrix representing the value of each state
        """
        # Embed hand and events.
        # event_embedding = self.event_embedding(event_history)
        hand_embedding = self.hand_embedding(player_info_state)

        # Pass events though gru, and hand through dense layer.
        gru_out, gru_state = self.critic_event_gru(inputs=event_history)
        dense_hand = self.critic_hand_dense(hand_embedding)

        # Concatonate and pass through dense layer.
        concatted = tf.concat(gru_out, dense_hand)
        dense_concatted = self.critic_concat_dense(concatted)

        # Return final value
        output = self.critic_output_dense(dense_concatted)
        return output

    def loss(self, event_history, player_info_state, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the handout to see how this is done.

        Remember that the loss is similar to the loss as in reinforce.py, with one specific change.

        1) Instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply
        with your advantage. Here, advantage is defined as discounted_rewards - state_values, where
        state_values is calculated by the critic network.

        2) In your actor loss, you must set advantage to be tf.stop_gradient(discounted_rewards -
        state_values). You may need to cast your (discounted_rewards - state_values) to tf.float32.
        tf.stop_gradient is used here to stop the loss calculated on the actor network from propagating
        back to the critic network.
        
        3) To calculate the loss for your critic network. Do this by calling the value_function on the
        states and then taking the sum of the squared advantage.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an
        [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as
        an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        
        prbs = tf.gather_nd(self.call(event_history, player_info_state),tf.convert_to_tensor(list(enumerate(actions))))
        advantages = tf.convert_to_tensor(discounted_rewards - self.value_function(event_history, player_info_state),dtype=tf.float32)
        stopped_advantages = tf.stop_gradient(advantages)
        weighted = tf.math.multiply(-tf.math.log(prbs),stopped_advantages)
        actor_loss = tf.math.reduce_sum(weighted)
        critic_loss = tf.math.reduce_sum(tf.math.square(advantages))
        return (actor_loss + critic_loss) / 2

