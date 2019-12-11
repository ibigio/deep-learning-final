import pyspiel
import numpy as np
from open_spiel.python import rl_environment
from itertools import permutations

import operator as op
from functools import reduce

game = 'liars_dice'
num_players = 2
num_dice = 5
num_faces = 6
dice_space = num_dice * num_faces
env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]


class LiarsDiceEnv:

  def __init__(self):
    self._env = rl_environment.Environment('liars_dice')
    self.num_players = 2
    self.num_faces = 6
    self.num_dice = 5
    self.num_actions = self._env.action_spec()['num_actions']

    self.liar_action_string = 'Liar'
    self.liar_action_id = 60
    self.wildcard = 6

    self.compute_translation_tables()


  def get_state(self):
    return self._env._state

  def reset(self):
    time_step = self._env.reset()
    self.decode_info_state(time_step.observations['info_state'])
    return time_step


  def step(self, action):
    time_step = self._env.step(action)
    self.decode_info_state(time_step.observations['info_state'])
    return time_step


  def compute_translation_tables(self):

    def binary_to_hand(binary_hand):
      return tuple([len(x) for x in ''.join([str(x) for x in binary_hand]).split('1')])

    def compute_all_encoded_dice_rolls():
      all_rolls = []
      for first in range(6):
        for second in range(6):
          for third in range(6):
            for fourth in range(6):
              for fifth in range(6):
                roll = [0] * (self.num_faces * self.num_dice)
                roll[(self.num_faces * 0) + first] = 1
                roll[(self.num_faces * 1) + second] = 1
                roll[(self.num_faces * 2) + third] = 1
                roll[(self.num_faces * 3) + fourth] = 1
                roll[(self.num_faces * 4) + fifth] = 1
                all_rolls.append(tuple(roll))
      return all_rolls

    binary_hands = sorted(set(permutations([0,1] * 5, 10)))
    self.id_to_unique_hand = [binary_to_hand(x) for x in binary_hands]
    self.unique_hand_to_id = {self.id_to_unique_hand[i]:i for i in range(len(self.id_to_unique_hand))}
    all_encoded_rolls = compute_all_encoded_dice_rolls()
    self.encoded_roll_to_id = {x:self.unique_hand_to_id[self.decode_roll_to_hand(x)] for x in all_encoded_rolls}
    self.hand_id_to_onehot = [self.hand_id_to_onehot(i) for i in range(len(self.id_to_unique_hand))]


  def hand_pretty(self, hand):
    return ''.join([str(i+1) * hand[i] for i in range(len(hand))])


  def hand_id_to_onehot(self, hand_id):
    onehot = [0] * len(self.id_to_unique_hand)
    onehot[hand_id] = 1
    return onehot


  def decode_roll_to_hand(self, encoded_roll):
    hand = [0] * self.num_faces
    for i in range(self.num_dice):
      index = i * self.num_faces
      face = np.argmax(encoded_roll[index:index + self.num_faces])
      hand[face] += 1
    return tuple(hand)


  def decode_player_info_state(self, player_info_state):
    dice_space = self.num_dice * self.num_faces
    encoded_roll_tuple = tuple(player_info_state[2:2+dice_space])
    hand_id = self.encoded_roll_to_id[encoded_roll_tuple]
    return hand_id


  def decode_info_state(self, info_state):
    info_state[0] = self.decode_player_info_state(info_state[0])
    info_state[1] = self.decode_player_info_state(info_state[1])


  def call_is_liar(self, call):
    if type(call) != str:
        return False
    return call.lower() == self.liar_action_string.lower()


  def action_id_to_call(self, action_id):
    quantity = (action_id // self.num_faces) + 1
    face = (action_id % self.num_faces) + 1
    if face == 0:
      face = self.num_faces
    if quantity > (self.num_players * self.num_dice):
      return self.liar_action_string
    return (quantity, face)


  def call_to_action_id(self, call):
    if self.call_is_liar(call):
        return self.liar_action_id
    quantity, face = call
    return ((quantity - 1) * num_faces) + (face - 1)

