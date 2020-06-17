import sys
import os
import torch
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from ddpg_agent import Agent

load_checkpoint_path = 'res/ddpg_64x32_2/'
training = False
if len(sys.argv) > 1:
    training = sys.argv[1] == "train"
    if not training and len(sys.argv) > 2:
        load_checkpoint_path = sys.argv[2]

if load_checkpoint_path[:-1] != '/':
    load_checkpoint_path += '/'

env = UnityEnvironment(file_name="tennis_win64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states)

agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, seed=21)


def save_hyper_params(f, a):
    f.write("DNN architecture, actor network:\n")
    f.write(str(a.actor_local))
    f.write("DNN architecture, critic network:\n")
    f.write(str(a.critic_local))
    f.write("\n\nTraining hyperparams:\n")
    f.write(a.get_hyper_params())
    f.write("\n\nTraining history:")
    f.flush()


def ddpg(res_path, n_episodes=20000, print_every=100):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    f = open(res_path + "train_history.txt", "w")
    plt_f = open(res_path + "plt_data.txt", "w")
    save_hyper_params(f, agent)
    t0 = time.perf_counter()
    total_scores = []  # list containing scores from each episode
    window_size = 100
    scores_window = deque(maxlen=window_size)  # last 100 scores
    sigma = 1.0

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)

        while True:
            actions = agent.act(states, sigma)

            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.all(dones):  # exit loop if episode finished
                break

        score = np.max(scores)
        scores_window.append(score)  # save most recent score
        average_score = np.mean(scores_window)
        total_scores.append(score)  # save most recent score
        sigma = min(1.0, 0.1 / (average_score + 0.01))

        t_elapsed = int(time.perf_counter() - t0)
        time_elapsed = "{}:{:02}".format(t_elapsed // 60, t_elapsed % 60)
        print('\rEpisode {}\tAverage Score: {:.2f}\tLast Score: {:.2f},{:.2f}\tTime elapsed: {}'.
              format(i_episode, average_score, scores[0], scores[1], time_elapsed), end="")
        plt_f.write('{} '.format(score))
        plt_f.flush()

        if i_episode % print_every == 0:
            msg = '\rEpisode {}.\tAverage Score: {:.2f}.\tTime elapsed: {}.'. \
                format(i_episode, np.mean(scores_window), time_elapsed)
            print(msg)
            f.write(msg)
            f.flush()
            torch.save(agent.actor_local.state_dict(), res_path + "actor_cp.pth")
            torch.save(agent.critic_local.state_dict(), res_path + "critic_cp.pth")
        if np.mean(scores_window) >= 0.5:
            msg = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}.\tTime elapsed: {}'. \
                format(i_episode - window_size, average_score, time_elapsed)
            print(msg)
            f.write(msg)
            break
    f.close()
    plt_f.close()
    return total_scores


if not training:
    agent.actor_local.load_state_dict(torch.load(load_checkpoint_path + "actor_cp.pth"))
    agent.critic_local.load_state_dict(torch.load(load_checkpoint_path + "critic_cp.pth"))

    for t in range(10):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            actions = agent.act(states, sigma=0.5)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        print('Total score (max over agents) this episode: {}'.format(np.max(scores)))
    # '''
else:
    training_num = 1
    while True:
        res_path = "res/{}/".format(training_num)
        try:
            os.makedirs(res_path)
            break
        except:
            training_num += 1

    scores = ddpg(res_path)

    torch.save(agent.actor_local.state_dict(), res_path + "actor_cp.pth")
    torch.save(agent.critic_local.state_dict(), res_path + "critic_cp.pth")

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(res_path + "score.png")
    plt.show()

env.close()
