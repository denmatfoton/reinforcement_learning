#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import random

from ppo_model import *

import torch
import torch.nn.functional as F
import torch.optim as optim

LR_ACTOR = 3e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        print("Using device: " + str(device))

        # Actor Network
        self.network = GaussianActorCriticNet(state_size, action_size,
                                              actor_body=FCBody(state_size, hidden_units=(256, 256), gate=torch.tanh),
                                              critic_body=FCBody(state_size, hidden_units=(256, 256), gate=torch.tanh))
        self.actor_optimizer = optim.Adam(self.network.actor_params, lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.network.critic_params, lr=LR_CRITIC)
        self.state_normalizer = MeanStdNormalizer()

        self.rollout_length = 128
        self.discount = 0.99
        self.gae_tau = 0.95
        self.optimization_epochs = 20
        self.mini_batch_size = 64
        self.entropy_weight = 0.
        self.target_kl = 0.01
        self.ppo_ratio_clip = 0.2

    def get_hyper_params(self):
        msg = ''
        msg += "lr_actor: {}\n".format(LR_ACTOR)
        msg += "lr_critic: {}\n".format(LR_CRITIC)
        msg += "rollout_length: {}\n".format(self.rollout_length)
        msg += "optimization_epochs: {}\n".format(self.optimization_epochs)
        msg += "mini_batch_size: {}\n".format(self.mini_batch_size)
        msg += "discount: {}\n".format(self.discount)
        msg += "gae_tau: {}\n".format(self.gae_tau)
        msg += "entropy_weight: {}\n".format(self.entropy_weight)
        msg += "target_kl: {}\n".format(self.target_kl)
        msg += "ppo_ratio_clip: {}\n".format(self.ppo_ratio_clip)
        return msg

    def play(self, env):
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(len(env_info.agents))  # initialize the score (for each agent)
        while True:
            states = self.state_normalizer(states)
            actions = to_np(self.network(states)['a'])  # select an action (for each agent)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            states = env_info.vector_observations  # get next state (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            if np.any(dones):  # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    # step for a set of agents
    def step(self, env):
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[brain_name]
        env_states = env_info.vector_observations
        num_agents = len(env_info.agents)
        scores = np.zeros(num_agents)

        loop = True
        while loop:
            storage = Storage(self.rollout_length)
            storage_size = 0

            for _ in range(self.rollout_length):
                states = self.state_normalizer(env_states)
                states = torch.from_numpy(states).float().to(device)
                prediction = self.network(states)
                env_info = env.step(to_np(prediction['a']))[brain_name]
                next_states = env_info.vector_observations  # get the next state
                rewards = env_info.rewards  # get the reward
                terminals = np.vstack(env_info.local_done).astype(np.uint8)  # see if episode has finished

                storage.add(prediction)
                storage.add({'r': tensor(rewards),
                             'm': tensor(1 - terminals),
                             's': tensor(states)})
                env_states = next_states
                scores += rewards
                storage_size += 1
                if np.all(terminals):
                    loop = False
                    break

            states = self.state_normalizer(env_states)
            states = torch.from_numpy(states).float().to(device)
            prediction = self.network(states)
            storage.add(prediction)
            storage.placeholder(storage_size)

            advantages = tensor(np.zeros((num_agents, 1)))
            returns = prediction['v'].detach()
            for i in reversed(range(storage_size)):
                returns = storage.r[i] + self.discount * storage.m[i] * returns
                # compute GAE
                td_error = storage.r[i] + self.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * self.gae_tau * self.discount * storage.m[i] + td_error
                storage.adv[i] = advantages.detach()
                storage.ret[i] = returns.detach()

            states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'], storage_size)
            actions = actions.detach()
            log_probs_old = log_probs_old.detach()
            advantages = (advantages - advantages.mean()) / advantages.std()
            self.learn(states, actions, log_probs_old, returns, advantages)

        return scores

    def learn(self, states, actions, log_probs_old, returns, advantages):
        storage_size = states.size(0)
        for _ in range(self.optimization_epochs):
            sampler = random_sample(np.arange(storage_size), self.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.ppo_ratio_clip,
                                          1.0 + self.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                if approx_kl <= 1.5 * self.target_kl:
                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self, size):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys, size):
        data = [getattr(self, k)[:size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)


class MeanStdNormalizer:
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def to_np(t):
    return t.cpu().detach().numpy()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]
