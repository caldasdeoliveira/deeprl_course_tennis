from agent import Agent

# +
from unityagents import UnityEnvironment
import numpy as np

import torch
# -

env = UnityEnvironment(file_name="Tennis.app", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# +
# Value for random seed
seed = 42

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
# -

agent = Agent(state_size, action_size, seed, use_prioritized_replay=False)

# +
from collections import namedtuple, deque

def ma_ddpg(n_episodes=2000, num_agents=2, window_len=100, goal = 0.5, print_every=1):
    assert num_agents >= 2, "There must be at least two agents for ma_ddpg"
    original_goal = goal
    scores_deque = deque(maxlen=window_len)
    scores = []
    avg_scores = []
    for i_episode in range(1, n_episodes+1):
        if i_episode % 1000 ==0:
            env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        else:
            env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                 # get the current state (for each agent)
        #agent.reset()
        score = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            
            actions = [agent.act(state.reshape([1,state.shape[0]])) for state in states]
            
            env_info = env.step(actions,)[brain_name]
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished 
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, i_episode,)
            states = next_states
            score += rewards                         # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        avg_scores.append(np.mean(scores_deque))
        if np.mean(scores_deque) >= goal and len(scores_deque) == window_len:
            if goal == original_goal:
                print(
                    f'\nEnvironment solved in {i_episode-100:d} episodes!'\
                    f'\tAverage Score: {np.mean(scores_deque):.3f}'
                )
            else:
                print(f"\nSaving better agent with Average Score: {np.mean(scores_deque):.3f}")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            goal = round(np.mean(scores_deque),1) + 0.1
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores, avg_scores


# -

scores,avg_scores = ma_ddpg(n_episodes=32000)

# +
import pickle

with open('scores.pkl', 'ab') as f:
    pickle.dump(scores,f) 

with open('avg_scores.pkl', 'ab') as f:
    pickle.dump(avg_scores,f)

# +
import matplotlib.pyplot as plt
# %matplotlib inline

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, 'b')
plt.plot(np.arange(len(scores)),avg_scores, 'r')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# -

env.close()


