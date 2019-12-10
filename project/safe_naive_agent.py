import operator as op
from functools import reduce
import numpy as np

num_players = 2
num_dice = 5
num_faces = 6
die_space = 5
dice_space = num_dice * die_space

LIAR_CALL = 'Liar'

"""
Helper functions.

    action_id_to_call:
        convert action id to call tuple

    call_to_action_id:
        convert call tuple to corresponding action id

    choose:
        implementatino of the "choose" function in probability

    player_info_state_to_hand:
        given an info_state, return our custom encoding of a hand
"""

def choose(n,k):
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return numer // denom

def player_info_state_to_hand(player_info_state):
    one_hot_hand = player_info_state[num_players:num_players+dice_space]
    numeric_hand = [0] * num_faces
    for i in range(num_dice):
        ind = i*die_space
        one_hot_die = one_hot_hand[ind:ind+die_space]
        if sum(one_hot_die) == 0:
            numeric_hand[5] += 1
        else:
            numeric_hand[np.argmax(one_hot_die)] += 1
    return numeric_hand

    

class SafeNaiveAgent:
    """
    Agent that uses naive bayes approach, looking only at the latest move
    and its own hand of dice.

    Selects and preforms highest probability move between:
        1. Making a call in the same quantity as previous.
        2. Making a call in a higher quantity than previous.
        3. Calling Liar.

    For optimization, this agent instanciates a probability lookup table
    upon initialization, to determine the probability of a given number
    of dice to be a certain value.
    """

    def __init__(self, env):
        self._env = env
        self.num_players = env.num_players
        self.num_faces = env.num_faces
        self.num_dice = env.num_dice
        self.wildcard = env.wildcard
        self.max_quantity = self.num_players * self.num_dice

        self.compute_probability_table()

    def compute_probability_table(self):
        self.probability_table = [[self.probability_of_at_least_helper(j, i==1) for j in range(self.max_quantity+1)] for i in range(2)]

    def probability_of_exactly(self, quantity, is_wildcard=False):
        # determine wildcard probability
        p = 2 / self.num_faces
        if is_wildcard:
            p = 1 / self.num_faces

        n = self.num_dice # number of unkown dice (in oponent's hand)
        k = quantity

        return choose(n,k) * (p ** k) * ((1-p) ** (n-k))


    def probability_of_at_least(self, quantity, is_wildcard=False):
        row = 1 if is_wildcard else 0
        return self.probability_table[row][quantity]

    def probability_of_at_least_helper(self, quantity, is_wildcard=False):
        total_probability = 0
        for i in range(quantity):
            total_probability += self.probability_of_exactly(i, is_wildcard)
        return 1 - total_probability
        

    def probability_of_call(self, quantity, face_value):
        real_quantity = quantity - self.hand[face_value - 1]

        # account for wildcards in our hand
        if face_value != self.wildcard:
            real_quantity -= self.hand[self.wildcard - 1]

        # our hand completely contains the call, so 100% likely
        if real_quantity < 1:
            return 1

        return self.probability_of_at_least(real_quantity, is_wildcard=(face_value==self.wildcard))


    def respond_to_call(self, quantity, face_value):

        same_quantity_call = None
        higher_quantity_call = None
        best_call = None
        same_quantity_call_probability = 0
        higher_quantity_call_probability = 0
        best_call_probability = 0

        # most likely call of same quantity (5 or less)
        if face_value < self.num_faces-1:
            # determine most common face value in our hand, greater than stated face_value
            most_likely_face_value = np.argmax(self.hand[face_value : self.num_faces-1]) + 1 + face_value
            same_quantity_call = (quantity, most_likely_face_value)

        # most likely call of higher quantity
        if quantity < self.max_quantity:
            most_likely_face_value = np.argmax(self.hand[:self.num_faces-1]) + 1
            higher_quantity_call = (quantity + 1, most_likely_face_value)

        # determine probabilities of each potential call
        if same_quantity_call is not None:
            same_quantity_call_probability = self.probability_of_call(*same_quantity_call)

        if higher_quantity_call is not None:
            higher_quantity_call_probability = self.probability_of_call(*higher_quantity_call)

        # determine best call between two options
        if same_quantity_call_probability > higher_quantity_call_probability:
            best_call = same_quantity_call
            best_call_probability = same_quantity_call_probability
        else:
            best_call = higher_quantity_call
            best_call_probability = higher_quantity_call_probability

        # determine whether to make call or call Liar
        incoming_call_probability = self.probability_of_call(quantity, face_value)

        # if our best call is more likely than calling Liar, return best call
        if best_call_probability > (1 - incoming_call_probability):
            return best_call
        # otherwise, return None (represents calling Liar)
        else:
            return LIAR_CALL

    def make_starting_call(self):

        starting_face_value = np.random.randint(6) + 1
        starting_quantity = self.hand[starting_face_value - 1]
        starting_quantity += max(1, np.random.randint(-1,2))

        return (starting_quantity, starting_face_value)

    def step(self, event_history, observations):
        """
        Determines safest call based on the previous call and its own hand.

        :param action_history: List of previous actions (ints), in order
        :param observations: Dict containing current state of game, with the following keys:
            - info_state        array [num_players x encoding_size] containing encoded game
                                state for each player
            - legal_actions     array [num_players x possible actions] containing available
                                legal actions for each player
            - current_player    int representing current player
        :return: An action (int)
        """
        cur_player = int(observations['current_player'])
        hand_id = np.argmax(observations['info_state'][cur_player][2:])
        self.hand = self._env.id_to_unique_hand[hand_id]

        # NOTE: this player will never call wildcards

        best_call = None

        # If it is responding to a call
        if event_history:
            last_event = event_history[-1] # only care about most recent event
            last_action = np.argmax(last_event[2:])
            call = self._env.action_id_to_call(last_action)
            quantity, face_value = call
            best_call = self.respond_to_call(quantity, face_value)
        
        # If it is starting
        else:
            best_call = self.make_starting_call()

        return self._env.call_to_action_id(best_call)
