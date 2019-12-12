import pyspiel
from pylab import *
import numpy as np
import tensorflow as tf

from model import ReinforceWithBaseline

from liars_dice_gym import LiarsDiceEnv
from safe_naive_agent import SafeNaiveAgent


def visualize_data(total_rewards):
    """
    Takes in array of rewards from each episode, visualizes reward over episodes.

    :param rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, 
    and returns a list of the sum of discounted rewards for
    each timestep. Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :return: discounted_rewards: list containing the sum of discounted rewards for each timestep in the original
    rewards list
    """
    # Compute discounted rewards (trust me this works and hopefully it's super fast)
    timesteps = len(rewards) # make into matrix
    rewards = tf.convert_to_tensor([rewards],dtype=tf.float32)
    # create lower triangular matrix of discount_factor weights
    T  = tf.convert_to_tensor([[max(1+i-j,0) for j in range(timesteps)] for i in range(timesteps)],dtype=tf.float32)
    T = tf.math.pow(discount_factor, T)
    T = tf.linalg.band_part(T, -1, 0)
    # apply discount factor
    return tf.matmul(rewards, T)




def generate_trajectory(env, model, adversary):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :return: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps
    in the episode
    """
    calls = []
    hands = []
    actions = []
    rewards = []

    time_step = env.reset()
    cur_agent, next_agent = model, adversary
    model_player_id = 0
    # TODO: add random starting

    last_call = None

    while not time_step.last():
        # get cur player id and hand
        cur_player_id = int(time_step.observations['current_player'])
        hand_id = time_step.observations['info_state'][cur_player_id]

        # If adversary's turn, make move and update last call
        if cur_player_id != model_player_id:
            action = adversary.step(last_call, hand_id)
            time_step = env.step([action])
            if time_step.last():
                rewards[-1] = time_step.rewards[model_player_id]
            last_call = action
            cur_agent, next_agent = next_agent, cur_agent
            continue

        # get action from agent
        if last_call == None:
            last_call = 1
        last_call_tensor = tf.convert_to_tensor([last_call], dtype=tf.float32)
        hand_id_tensor = tf.convert_to_tensor([hand_id], dtype=tf.float32)
        prbs = cur_agent.call(last_call_tensor, hand_id_tensor)[0].numpy()

        # mask out illegal actions
        legal_actions = time_step.observations['legal_actions'][cur_player_id]
        legal_actions_mask = np.ones(env.num_actions, dtype=bool)
        legal_actions_mask[legal_actions] = False
        prbs[legal_actions_mask] = 0

        # renormalize probabilities
        norm = np.sum(prbs)
        # TODO: check for zero norm
        if norm == 0:
            old_prbs = prbs
            prbs = np.zeros(env.num_actions)
            prbs[legal_actions] += (1/len(legal_actions))
            if np.isnan(prbs).any():
                print('Before:', old_prbs)
                print('Legal Actions:', legal_actions)
                print('After:', prbs)
        else:
            old_prbs = prbs

            prbs = prbs / norm

            if np.isnan(prbs).any():
                print('Before:',old_prbs)
                print('Norm:',norm)
                print('After:', prbs)
                exit()


        # select action weighted by prbs
        action = np.random.choice(list(range(len(prbs))), p=prbs)
        # apply action to env
        time_step = env.step([action])

        # update calls, hands, actions, and rewards
        calls.append(last_call)
        hands.append(hand_id)
        actions.append(action)
        rewards.append(time_step.rewards[cur_player_id])

        last_call = action
        cur_agent, next_agent = next_agent, cur_agent

    return calls, hands, actions, rewards


def train(env, model, adversary):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
    and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode.

    :param env: The openai gym environment
    :param model: The model
    :return: The total reward for the episode
    """

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.

    with tf.GradientTape() as tape:
        calls, hands, actions, rewards = generate_trajectory(env, model, adversary)
        discounted = discount(rewards)
        loss = model.loss(np.array(calls), np.array(hands), np.array(actions), discounted)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.sum(rewards)



def main():

    env = LiarsDiceEnv()
    num_actions = env.num_actions

    # Initialize model
    model = ReinforceWithBaseline(num_actions)
    adversary = SafeNaiveAgent(env)

    # TODO: 
    # 1) Train your model for 650 episodes, passing in the environment and the agent. 
    all_rewards = []
    epochs = 10000
    for i in range(epochs):
        all_rewards.append(train(env, model, adversary))
        if i % 50 == 0:
            print("Reward of past 50:",np.mean(all_rewards[-50:]))
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
    # 3) After training, print the average of the last 50 rewards you've collected.

    # TODO: Visualize your rewards.
    # visualize_data(all_rewards)


if __name__ == '__main__':
    main()

