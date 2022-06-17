# Udacity Deep Reinforcement Learning course - Project 3: Tennis

![]()

This repository contains my solution for the third project of Udacity's course on Reinforcement Learning. The scenario's goal is to !!!TODO!!!!
## Contents
This repo contains:
!!!TODO!!!!

## Getting Started

This project was developed and tested on an Apple Macbook with an intel i5. No guarantees are made on performance in other systems. 

### Mac

To setup your coding evironment you need to perform 3 steps after cloning this repository:

1. Make `setup.sh` executable. There are many ways to do this. One way is through the terminal run this command:

```bash
chmod +x setup.sh
```

2. Then you simply run `setup.sh`.

3. Finally you activate the conda environment in your terminal or on your notebook change the kernel to `drl_tennis`
### Others

If you are running this on other operating systems. There's a strong possibility that you can just follow the instructions for mac. Otherwise you will need to follow the steps in the [readme](Value-based-methods/README.md) in the `Value-based-methods` repo.

#### To download the environment for other Operating Systems
If you are using anothe OS you'll need to manually download the environment from one of the links below.  You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## The Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Training and report

The training code can be found in `train.py`. This is a jupyter notebook created with [jupytext](https://github.com/mwouts/jupytext#:~:text=Jupytext%20is%20a%20plugin%20for,Scripts%20in%20many%20languages.) so it can be opened either as a notebook or used as a script. 

The final report can be found in `report.ipynb` and is a regular notebook as required for the delivery of the project.

## Author

Diogo Oliveira