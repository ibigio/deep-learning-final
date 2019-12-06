import pyspiel
import numpy as np
from open_spiel.python import rl_environment

from safe_naive_agent import SafeNaiveAgent
from random_agent import RandomAgent
from interactive_agent import InteractiveAgent

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

def hand_pretty(hand):
    return ''.join([str(i+1) * hand[i] for i in range(len(hand))])

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

def play_game_verbose(env, agent_1, agent_2):
  cur_agent, next_agent = agent_1, agent_2
  action_history = []

  time_step = env.reset()
  while not time_step.last():
    # get action from agent
    action = cur_agent.step(action_history, time_step.observations)
    # apply action to env
    time_step = env.step([action])
    # update actions
    action_history.append(action)
    # update current agent
    cur_agent, next_agent = next_agent, cur_agent
  for i in range(num_players):
    pretty_hand = hand_pretty(player_info_state_to_hand(time_step.observations['info_state'][i]))
    print(f"Player {i}'s Hand: {pretty_hand}")
  print(f'Winner: Player {np.argmax(time_step.rewards)}')
  return time_step.rewards

def play_game(env, agent_1, agent_2):
  cur_agent, next_agent = agent_1, agent_2
  action_history = []

  time_step = env.reset()
  while not time_step.last():
    # get action from agent
    action = cur_agent.step(action_history, time_step.observations)
    # apply action to env
    time_step = env.step([action])
    # update actions
    action_history.append(action)
    # update current agent
    cur_agent, next_agent = next_agent, cur_agent
  return time_step.rewards


safe_agent = SafeNaiveAgent()
random_agent = RandomAgent()
interactive_agent = InteractiveAgent()

# play_game_verbose(env, interactive_agent, safe_agent)

# exit()

num_games = 10000
p0_wins = 0
for i in range(num_games):
  if i % 1000 == 0:
    print(i)
  rewards = play_game(env, safe_agent, safe_agent)
  if rewards[0] > 0:
    p0_wins += 1

print('Agent Win Rate:', 1-(p0_wins/num_games))

