import pyspiel
import numpy as np
from open_spiel.python import rl_environment

from safe_naive_agent import SafeNaiveAgent
from random_agent import RandomAgent

import operator as op
from functools import reduce

game = 'liars_dice'
num_players = 2
num_dice = 5
num_faces = 6
dice_space = num_dice * num_faces
env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]

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

num_games = 10000
p0_wins = 0
for i in range(num_games):
  if i % 1000 == 0:
    print(i)
  rewards = play_game(env, random_agent, safe_agent)
  if rewards[0] > 0:
    p0_wins += 1

print('Agent Win Rate:', 1-(p0_wins/num_games))

