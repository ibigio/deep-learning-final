import pyspiel
import numpy as np
import tensorflow as tf
from open_spiel.python import rl_environment

from safe_naive_agent import SafeNaiveAgent
from random_agent import RandomAgent
from model import ReinforceAgent

import operator as op
from functools import reduce

def discount(prbs):
    # TODO: use Ilan's amazing discount
    return prbs

def play_game(env, other_agent, trainable_agent):
    cur_agent, next_agent = trainable_agent, other_agent
    action_history = []

    time_step = env.reset()
    while not time_step.last():
        # get action from agent
        if cur_agent == trainable_agent:
            with tf.GradientTape() as tape:
                prbs = cur_agent.step(action_history, time_step.observations)
                discounted = discount(prbs)
                loss = cur_agent.loss(prbs, action_history, discounted)

            gradients = tape.gradient(loss, cur_agent.trainable_variables)
            cur_agent.optimizer.apply_gradients(zip(gradients, cur_agent.trainable_variables))

            action = np.random.choice()
        else:
            action = cur_agent.step(action_history, time_step.observations)
        # apply action to env
        time_step = env.step([action])
        # update actions
        action_history.append(action)
        # update current agent
        cur_agent, next_agent = next_agent, cur_agent

    return time_step.rewards

def main():
    game = 'liars_dice'
    num_players = 2
    num_dice = 5
    num_faces = 6
    dice_space = num_dice * num_faces
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    safe_agent = SafeNaiveAgent()
    reinforce_agent = ReinforceAgent()

    num_games = 10000
    p0_wins = 0
    for i in range(num_games):
        if i % 1000 == 0:
            print(i)
        rewards = play_game(env, safe_agent, reinforce_agent)
        if rewards[0] > 0:
            p0_wins += 1

    print('Agent Win Rate:', 1-(p0_wins/num_games))

if __name__ == "__main__":
    main()