# Continuous Control in Reacher Environment

This project is intended to solve [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
Environment. This project was created as part of
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) course.
All of the code is in PyTorch (v1.5) and Python 3.

![Unity Reacher environment](https://github.com/Unity-Technologies/ml-agents/raw/master/docs/images/reacher.png)

## Environment
In this environment, a double-jointed arm can move to target locations.
A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular
velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints.
Every entry in the action vector should be a number between -1 and 1.

For this project two separate versions of the Unity environment were provided:

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

###Solving the Environment

####Solving the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score
of +30 over 100 consecutive episodes.

####Solving the Second Version
The barrier for solving the second version of the environment is slightly different, to take into
account the presence of many agents. In particular, your agents must get an average score
of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).

## Getting started
Download the whole repository.

### Packages
You should have installed Python 3.8 as well. Create new python virtual environment. Install required python packages in this virtual environment:
1. Pytorch (1.5.0)
2. numpy
3. matplotlib
From the root of this repo run `pip -q install ./python`. This will install packages required for connection with the environment.

### Unity Environment
The repository already contains a prebuilt environment for Windows (64-bit).

If you have another OS, download the environment from one of the links below and replace:

####Version 1: One (1) Agent
 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
 - Max OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip) 
 
####Version 2: Twenty (20) Agents
 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
 - Max OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip) 

### Run
Train: `python robot_arm.py train`

Evaluate: `python robot_arm.py test <checkpoint>`

## Results

TODO: Add results
 
 ## Additional info
 
 See report for additional information.
 