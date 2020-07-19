import operator as op
from functools import reduce
import numpy as np
import re

num_players = 2
num_dice = 5
num_faces = 6
die_space = 5
dice_space = num_dice * die_space

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

def input_call(text):
    raw = input(text)
    if raw == 'Liar':
        return 'Liar'
    call_array = re.findall(r'\d+',raw)
    return tuple([int(x) for x in call_array])

def hand_pretty(hand):
    return ''.join([str(i+1) * hand[i] for i in range(len(hand))])

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

    

class InteractiveAgent:
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

    def __init__(self):
        pass

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
        cur_player = int(observations['current_player'])
        self.hand = player_info_state_to_hand(observations['info_state'][cur_player]) # hand for the round (a 4 and three 5s has form [0,0,0,1,3,0])
        print('Hand:', hand_pretty(self.hand))
        if action_history:
            last_call = action_id_to_call(action_history[-1])
            print("Opponent's Call:", last_call)
        else:
            print("You start!")

        call = input_call('Your Action:')

        return call_to_action_id(call)
