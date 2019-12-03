import random
import pyspiel
import numpy as np
from open_spiel.python import rl_environment

import operator as op
from functools import reduce

game = 'liars_dice'
num_players = 2
num_dice = 5
num_faces = 6
dice_space = num_dice * num_faces
env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]

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

# def prob_exactly(n, k, p):
#     return choose(n,k) * (p ** k) * ((1-p) ** (n-k))

class SafeNaiveAgent:

  def __init__(self, hand):
    self.hand = hand # hand for the round (a 4 and three 5s has form [0,0,0,1,3,0])
    self.num_faces = 6
    self.wildcard = 6
    self.num_dice = 5
    self.max_quantity = 10

  def set_hand(self, hand):
    self.hand = hand

  def probability_of_exactly(self, quantity, is_wildcard=False):
    # determine wildcard probability
    p = 2 / self.num_faces
    if is_wildcard:
      p = 1 / self.num_faces

    n = self.num_dice # number of unkown dice (in oponent's hand)
    k = quantity

    return choose(n,k) * (p ** k) * ((1-p) ** (n-k))


  def probability_of_at_least(self, quantity, is_wildcard=False):
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


  def step(self, states):
    last_action = states[-1] # only care about most recent call

    call = action_id_to_call(last_action)
    quantity, face_value = call

    # NOTE: this player will never call wildcards

    best_call = self.respond_to_call(quantity, face_value)

    # TODO: hard coded for 2 players, 5 dice each!
    # total_calls = 61
    # prbs = [0] * total_calls
    # prbs[call_to_action_id(best_call)] = 1
    # return prbs

    return call_to_action_id(best_call)



def hand_to_string(hand):
  faces = 6
  numeric_hand = []
  for i in range(faces):
    ind = i*faces
    if hand[ind:ind+faces]:
      numeric_hand.append(str(np.argmax(hand[ind:ind+faces]) + 1))
  return ''.join(numeric_hand)

def info_state_to_hand(info_state):
  faces = 6
  one_hot_hand = info_state[num_players:num_players+dice_space]
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

def play_game(env):
  time_step = env.reset()
  for player_id in range(num_players):
    print(f"Player {player_id} Hand: {hand_to_string(time_step.observations['info_state'][player_id][num_players:num_players+dice_space])}")
    print(f"Hand {info_state_to_hand(time_step.observations['info_state'][player_id])}")
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    actions = time_step.observations['legal_actions'][player_id]
    action = np.random.choice(actions)
    print(f'Player {player_id}: {env._state.action_to_string(action)} ({action})')

    time_step = env.step([action])
  print(time_step)
  return time_step.rewards

def play_game_safe_vs_random(env):
  time_step = env.reset()

  # print player hands
  # for player_id in range(num_players):
  #   print(f"Player {player_id} Hand: {hand_to_string(time_step.observations['info_state'][player_id][num_players:num_players+dice_space])}")
  
  hand = info_state_to_hand(time_step.observations['info_state'][1])
  safe_agent = SafeNaiveAgent(hand)

  state_list = []
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    action = None
    if player_id == 1:
      action = safe_agent.step(state_list)
    else:
      actions = time_step.observations['legal_actions'][player_id]
      action = np.random.choice(actions)
    state_list.append(action)
    # print(f'Player {player_id}: {env._state.action_to_string(action)} ({action})')

    time_step = env.step([action])
  # print(time_step)
  return time_step.rewards


num_games = 10000
p0_wins = 0
for i in range(num_games):
  if i % 1000 == 0:
    print(i)
  rewards = play_game_safe_vs_random(env)
  if rewards[0] > 0:
    p0_wins += 1

print('Agent Win Rate:', 1-(p0_wins/num_games))


# player = SafeNaiveAgent([1,0,4,0,0,0])
# for q in range(1,10):
#   for fv in range(1,6):
#     call = player.step([(q,fv)])
#     action = np.argmax(call)
#     print(f"Responded {action_id_to_call(action)} to call ({q}-{fv})")




exit()

# num_games = 10000
# p0_wins = 0
# for i in range(num_games):
#   if i % 1000 == 0:
#     print(i)
#   rewards = play_game(env)
#   if rewards[0] > 0:
#     p0_wins += 1

# print('Win Rate:', p0_wins/num_games)


# print(time_step.observations['info_state'][0])
# print(time_step.observations['info_state'][1])
# print(time_step)
# exit()

game = pyspiel.load_game("liars_dice")
state = game.new_initial_state()
while not state.is_terminal():
  legal_actions = state.legal_actions()
  # print([state.action_to_string(a) for a in legal_actions])
  print(state)
  if state.is_chance_node():
    # Sample a chance event outcome.
    outcomes_with_probs = state.chance_outcomes()
    # print(outcomes_with_probs)
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(action)
  else:
    # The algorithm can pick an action based on an observation (fully observable
    # games) or an information state (information available for that player)
    # We arbitrarily select the first available action as an example.
    # action = legal_actions[0]
    action = np.random.choice(legal_actions)
    print(state.action_to_string(action))
    state.apply_action(action)

print(state.returns())

