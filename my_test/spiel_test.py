import random
import pyspiel
import numpy as np
from open_spiel.python import rl_environment


game = 'liars_dice'
num_players = 2
env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]

time_step = env.reset()
print(time_step.observations['info_state'][0])
print(time_step.observations['info_state'][1])
print(time_step)
exit()

game = pyspiel.load_game("liars_dice")
state = game.new_initial_state()
while not state.is_terminal():
  legal_actions = state.legal_actions()
  print([state.action_to_string(a) for a in legal_actions])
  if state.is_chance_node():
    # Sample a chance event outcome.
    outcomes_with_probs = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(action)
  else:
    # The algorithm can pick an action based on an observation (fully observable
    # games) or an information state (information available for that player)
    # We arbitrarily select the first available action as an example.
    action = legal_actions[0]
    state.apply_action(action)

