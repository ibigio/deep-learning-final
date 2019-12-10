import pyspiel
import numpy as np

from liars_dice_gym import LiarsDiceEnv
from safe_naive_agent import SafeNaiveAgent

def play_game_verbose(env, agent_1, agent_2):
  cur_agent, next_agent = agent_1, agent_2
  event_size = 63
  event_history = []

  time_step = env.reset()
  while not time_step.last():
    cur_player_id = int(time_step.observations['current_player'])
    # get action from agent
    action = cur_agent.step(event_history, time_step.observations)
    # apply action to env
    time_step = env.step([action])
    # update actions
    event = np.zeros(event_size)
    event[cur_player_id] = 1
    event[2 + action] = 1
    event_history.append(event)
    # update current agent
    cur_agent, next_agent = next_agent, cur_agent
  for i in range(env.num_players):
    pretty_hand = env.hand_pretty(env.id_to_unique_hand[np.argmax(time_step.observations['info_state'][i])])
    print(f"Player {i}'s Hand: {pretty_hand}")
  print(f'Winner: Player {np.argmax(time_step.rewards)}')
  print(time_step)
  print(env.get_state())
  return time_step.rewards

env = LiarsDiceEnv()
safe_agent = SafeNaiveAgent(env)

play_game_verbose(env, safe_agent, safe_agent)