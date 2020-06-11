import sys
import os
import torch
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from ppo_agent import Agent


load_checkpoint_path = ''
training = False
if len(sys.argv) > 1:
    training = sys.argv[1] == "train"
    if not training and len(sys.argv) > 2:
        load_checkpoint_path = sys.argv[2]

if load_checkpoint_path[:-1] != '/':
    load_checkpoint_path += '/'


# please do not modify the line below
env = UnityEnvironment(file_name="20_arms_win64/Reacher.exe")

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


agent = Agent(state_size=state_size, action_size=action_size, seed=21)


def save_hyper_params(f, agent):
    f.write("DNN architecture:\n")
    f.write(str(agent.network))
    f.write("\n\nTraining hyperparams:\n")
    f.write(agent.get_hyper_params())
    f.write("\n\nTraining history:")
    f.flush()


def ppo(res_path, n_episodes=2000, print_every=10):
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

    for i_episode in range(1, n_episodes + 1):
        scores = agent.step(env)

        score = np.mean(scores)
        scores_window.append(score)  # save most recent score
        average_score = np.mean(scores_window)
        total_scores.append(score)  # save most recent score

        t_elapsed = int(time.perf_counter() - t0)
        time_elapsed = "{}:{:02}".format(t_elapsed // 60, t_elapsed % 60)
        print('\rEpisode {}\tAverage Score: {:.2f}\tLast Score: {:.2f}\tTime elapsed: {}'.
              format(i_episode, average_score, score, time_elapsed), end="")
        plt_f.write('{} '.format(score))
        plt_f.flush()

        if i_episode % print_every == 0:
            msg = '\rEpisode {}\tAverage Score: {:.2f}\tTime elapsed: {}'.\
                format(i_episode, np.mean(scores_window), time_elapsed)
            print(msg)
            f.write(msg)
            f.flush()
            torch.save(agent.network.state_dict(), res_path + "cp.pth")
            torch.save(agent.network.actor_body.state_dict(), res_path + "actor_cp.pth")
            torch.save(agent.network.critic_body.state_dict(), res_path + "critic_cp.pth")
        if np.mean(scores_window) >= 30.0:
            msg = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime elapsed: {}s'.\
                format(i_episode - window_size, average_score, time_elapsed)
            print(msg)
            f.write(msg)
            break
    f.close()
    plt_f.close()
    return total_scores


if not training:
    agent.network.load_state_dict(torch.load(load_checkpoint_path + "cp.pth"))

    for t in range(10):
        agent.play(env)

else:
    training_num = 1
    while True:
        res_path = "res/{}/".format(training_num)
        try:
            os.makedirs(res_path)
            break
        except:
            training_num += 1

    plot_scores = ppo(res_path)

    torch.save(agent.network.state_dict(), res_path + "cp.pth")
    torch.save(agent.network.actor_body.state_dict(), res_path + "actor_cp.pth")
    torch.save(agent.network.critic_body.state_dict(), res_path + "critic_cp.pth")

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(plot_scores)), plot_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(res_path + "score.png")
    plt.show()

env.close()
