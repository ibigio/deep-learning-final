import operator as op
from functools import reduce
import numpy as np

num_players = 2
num_dice = 5
num_faces = 6
dice_space = num_dice * num_faces

def action_id_to_call(bidnum):
    quantity = (bidnum // num_faces) + 1
    face_value = 1 + (bidnum % num_faces)
    if face_value == 0:
        face_value = num_faces
    if quantity > (num_players * num_dice):
        return 'Liar'
    return (quantity, face_value)

def call_to_action_id(call):
    if call == 'Liar':
        return 60
    quantity, face_value = call
    return ((quantity - 1) * num_faces) + (face_value - 1)

def choose(n,k):
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return numer // denom

def player_info_state_to_hand(player_info_state):
    faces = 6
    one_hot_hand = player_info_state[num_players:num_players+dice_space]
    numeric_hand = [0] * faces
    for i in range(faces):
        ind = i*faces
        one_hot_die = one_hot_hand[ind:ind+faces]
        if not one_hot_die: # end of list of dice
            continue
        if sum(one_hot_die) == 0:
            numeric_hand[5] += 1
        else:
            numeric_hand[np.argmax(one_hot_die)] += 1
    return numeric_hand

class SafeNaiveAgent:

    def __init__(self):
        self.num_faces = 6
        self.wildcard = 6
        self.num_dice = 5
        self.max_quantity = 10
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
            return 'Liar'

    def step(self, action_history, observations):
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
        last_action = action_history[-1] # only care about most recent call
        cur_player = int(observations['current_player'])

        self.hand = player_info_state_to_hand(observations['info_state'][cur_player]) # hand for the round (a 4 and three 5s has form [0,0,0,1,3,0])

        call = action_id_to_call(last_action)
        quantity, face_value = call

        # NOTE: this player will never call wildcards
        best_call = self.respond_to_call(quantity, face_value)

        return call_to_action_id(best_call)

