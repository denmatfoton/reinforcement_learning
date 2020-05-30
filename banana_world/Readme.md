# Banana World Double DQN Agent

This project is intended to solve BananaWorld Unity Environment. This project was
 created as part of [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) course.
All of the code is in PyTorch (v1.5) and Python 3.

![Unity Banana environment](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

**NOTE:**
* The environment provided by Udacity is similar to, but **not identical to** the Banana Collector environment on the [Unity ML-Agents Github page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). 

## Environment
The **state space** has `37` dimensions each of which is a continuous variable. It includes the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
The **action space** contains the following `4` legal actions: 
- move forward `0`
- move backward `1`
- turn left `2`
- turn right `3`

A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The task is **episodic**, and in order to solve the environment, your agent must get an average score of `+13` over `100` consecutive episodes.

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
 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - Max OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) 

### Run
Train: `python banana_world.py train`

Evaluate: `python banana_world.py test <checkpoint>`

## Results

By using Double DQN architecture the agent was able to solve the environment after 118 episodes!
 See the video below.

[![Agent demonstration](https://img.youtube.com/vi/PPSJ2k9RBq0/0.jpg)](https://youtu.be/PPSJ2k9RBq0)
 
 ## Additional info
 
 See report for additional information.
 