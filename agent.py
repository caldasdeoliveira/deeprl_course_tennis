import numpy as np
import random
from collections import namedtuple, deque
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from agent_utils.ounoise import OUNoise
from agent_utils.replaybuffer import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self,
     state_size,
     action_size,
     seed,
     model_state_dict_path = None, # enables loading of a trained agent
     buffer_size = int(2**22),
     batch_size = 128,
     gamma = 0.99,
     tau = 1e-3,
     lr_actor = 1e-3,
     lr_critic = 1e-3,
     learning_passes = 5,           # number of learning passes
     noise_decay = 0.99,
     update_every = 2**3,
     ):
        """ DDPG agent
        This class instantiates a DDPG agent with TODO
        Args:
            state_size (int): shape of the state encoding.
            action_size (int): number of actions in the environment.
            seed (int): seed for random number generators.
            model_state_dict_path (str): path to 
            buffer_size (int): "memory size", how many experiences can the agent store.
            batch_size (int): number of experiences in a batch.
            gamma (float): reward decay, rewards in the future are worse than
                           imidiate rewards. Between 0 and 1. 
            tau (float): factor with which we control the size of the update
                         of the target network. Between 0 and 1.
            lr_actor (float): learning rate for the actor network. Between 0 and 1.
            lr_critic (float): learning rate for the critic network. Between 0 and 1.
            noise_decay (int): TODO
            update_every (int): number of episodes after which the agent target network weights 
                                are updated.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.learning_passes = learning_passes
        self.noise_factor = 1
        self.noise_decay = noise_decay
        self.update_every = update_every
        self.last_episode_train = 0


        # Actor Network
        self.actor_local = Actor(state_size, action_size, seed,).to(device)
        self.actor_target = Actor(state_size, action_size, seed,).to(device)
        if model_state_dict_path:
            model_path = os.path.join(model_state_dict_path,"checkpoint_actor.pth")
            print(f"Loading actor from: {model_path}")
            self.actor_local.load_state_dict(torch.load(model_path))
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, seed,).to(device)
        self.critic_target = Critic(state_size, action_size, seed,).to(device)
        if model_state_dict_path:
            model_path = os.path.join(model_state_dict_path,"checkpoint_critic.pth")
            print(f"Loading critic from: {model_path}")
            self.critic_local.load_state_dict(torch.load(model_path))
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)

        # Initialize target networks weights with the local networks ones
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(action_size, seed, buffer_size=buffer_size, batch_size=batch_size)

        # Noise process
        self.noise = OUNoise(action_size, seed)

    def act(self, state, noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval() # setting to eval mode
        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train() # setting to train mode

        #print(f"pre noise action {action}")
        if noise:
            # Add noise to the action in order to explore the environment
            action += self.noise_factor * self.noise.sample()

        #print(f"post noise action {action}")

        return np.clip(action, -1, 1)
        
    def step(self, states, actions, rewards, next_states, dones, episode_number=0):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        # Save experience
        self.replay_buffer.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > self.batch_size:
            if episode_number % self.update_every == 0 and self.last_episode_train != episode_number:
                for _ in range(self.learning_passes):
                    experiences = self.replay_buffer.sample()
                    self.learn(experiences)
                
                self.last_episode_train = episode_number
                
                # Decay the noise process
                self.noise_factor *= self.noise_decay
                self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions from actor_target model
        actions_next = self.actor_target(next_states)
        # Get predicted next-state Q-Values from critic_target model
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) TODO
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model, tau=1e-3):
        """ Soft update model parameters.
        
        Update using the following formula
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
