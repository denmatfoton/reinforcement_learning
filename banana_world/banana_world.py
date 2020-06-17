import sys
import os
import torch
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from dqn_agent import Agent

load_checkpoint = 'res/DDQN_64x32/checkpoint.pth'
training = False
if len(sys.argv) > 1:
    training = sys.argv[1] == "train"
    if not training and len(sys.argv) > 2:
        load_checkpoint = sys.argv[2]

# please do not modify the line below
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)
print('State size:', state_size)

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

def save_hyper_params(f, agent, eps_start, eps_end, eps_decay):
    f.write("DNN architecture:\n")
    f.write(str(agent.qnetwork_local))
    f.write("\n\nTraining hyperparams:\n")
    f.write("eps_start: {}, eps_end: {}, eps_decay: {}\n".format(eps_start, eps_end, eps_decay))
    f.write(agent.get_hyper_params())
    f.write("\n\nTraining history:")
    f.flush()


def dqn(res_path, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.97, print_every=10):
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
    save_hyper_params(f, agent, eps_start, eps_end, eps_decay)
    t0 = time.perf_counter()
    scores = []  # list containing scores from each episode
    window_size = 100
    scores_window = deque(maxlen=window_size)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        for t in range(max_t):
            action = int(agent.act(state, eps))
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        average_score = np.mean(scores_window)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        t_elapsed = int(time.perf_counter() - t0)
        time_elapsed = "{}:{:02}".format(t_elapsed // 60, t_elapsed % 60)
        print('\rEpisode {}\tAverage Score: {:.2f}\tLast Score: {:.2f}\tTime elapsed: {}'.
              format(i_episode, average_score, score, time_elapsed), end="")
        plt_f.write('{} '.format(score))
        plt_f.flush()

        if i_episode % print_every == 0:
            msg = '\rEpisode {}.\tAverage Score: {:.2f}.\tTime elapsed: {}.'. \
                format(i_episode, np.mean(scores_window), time_elapsed)
            print(msg)
            f.write(msg)
            f.flush()
            torch.save(agent.qnetwork_local.state_dict(), res_path + "checkpoint.pth")
        if np.mean(scores_window) >= 13.0:
            msg = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}.\tTime elapsed: {}'. \
                format(i_episode - window_size, average_score, time_elapsed)
            print(msg)
            f.write(msg)
            break
    f.close()
    plt_f.close()
    return scores


if not training:
    agent.qnetwork_local.load_state_dict(torch.load(load_checkpoint))
    for t in range(10):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = int(agent.act(state))  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        print("Score: {}".format(score))
else:
    training_num = 1
    while True:
        res_path = "res/{}/".format(training_num)
        try:
            os.makedirs(res_path)
            break
        except:
            training_num += 1

    scores = dqn(res_path)

    torch.save(agent.qnetwork_local.state_dict(), res_path + "checkpoint.pth")

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(res_path + "score.png")
    plt.show()

env.close()
