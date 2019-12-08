import tensorflow as tf
from tf.keras.layers import Embedding, GRU, Dense
import pyspiel
from open_spiel.python import rl_environment

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ReinforceAgent(tf.keras.Model):
    def __init__(self, num_actions):
        super(ReinforceAgent, self).__init__()
        self.num_actions = num_actions

        self.start_embedding_size = 50
        self.gru_embedding_size = 100
        self.hidden_dense_size = 100

        self.state_embedding = Embedding(63, self.start_embedding_size)
        self.hand_embedding = Embedding(252, self.start_embedding_size)
        self.gru = GRU(self.gru_embedding_size, return_sequences=True, return_state=True)
        self.hand_dense_layer = Dense(self.hidden_dense_size)
        self.concatted_dense_1 = Dense(self.hidden_dense_size)
        self.final_dense = Dense(61, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def step(self, action_history, observations):
        """
        Determines safest call by calling this model.
        """
        return self.call(action_history, observations)

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
        states = tf.convert_to_tensor(states)
        hand = tf.convert_to_tensor(hand)
        # 1. Raise each state in states with an embedding layer.
        state_embedding = self.state_embedding(states)
        # 2. Raise the hand with a different embedding layer.
        hand_embedding = self.hand_embedding(hand)
        # 3. Feed states through a GRU, and get the last hidden state.
        gru_out, gru_state = self.gru(inputs=state_embedding)
        # 4. Feed hand through a Dense layer.
        dense_hand = self.hand_dense_layer(hand_embedding)
        # 5. Concat output of GRU and output of Dense layer.
        concatted = tf.concat(gru_out, dense_hand)
        # 6. Feed concat output through one (or more) Dense layer(s).
        concatted_dense_1 = self.concatted_dense_1(concatted)
        # 7. Feed output through Dense layer to size num_actions (61), and call softmax.
        final_dense = self.final_dense(concatted_dense_1)
        # 8 TODO: gotta be train/test I think (maybe here, or maybe in train/test). Mask out illegal calls, and re-normalize.
        # 9. Return matrix of probability distribution over actions.
        return final_dense

    def loss(self, probabilities, actions, discounted_rewards):
        gathered = tf.gather_nd(probabilities, [[ind, action] for ind, action in enumerate(actions)])
        neg = -tf.math.log(gathered)
        weighted = neg * discounted_rewards
        return tf.reduce_sum(weighted)