import pyspiel
import numpy as np

from liars_dice_gym import LiarsDiceEnv
from safe_naive_agent import SafeNaiveAgent

def play_game_verbose(env, agent_1, agent_2):
  cur_agent, next_agent = agent_1, agent_2

  last_action = None

  time_step = env.reset()
  while not time_step.last():
    # get current player id
    cur_player_id = int(time_step.observations['current_player'])

    # get action from agent
    hand_id = time_step.observations['info_state'][cur_player_id]
    action = cur_agent.call(last_action, hand_id)


    # apply action to env
    time_step = env.step([action])

    # update current agent
    last_action = action
    cur_agent, next_agent = next_agent, cur_agent
  for i in range(env.num_players):
    pretty_hand = env.hand_pretty(env.id_to_unique_hand[time_step.observations['info_state'][i]])
    print(f"Player {i}'s Hand: {pretty_hand}")
  print(f'Winner: Player {np.argmax(time_step.rewards)}')
  print(time_step)
  print(env.get_state())
  return time_step.rewards

env = LiarsDiceEnv()
safe_agent = SafeNaiveAgent(env)

play_game_verbose(env, safe_agent, safe_agent)