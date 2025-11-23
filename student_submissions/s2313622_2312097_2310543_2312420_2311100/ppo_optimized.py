# ACKNOWLEDGEMENT: This code is based on the PPO implementation from the following source:
# https://github.com/ericyangyu/PPO-for-Beginners

import argparse
import csv
import gym_cutting_stock
import gymnasium as gym
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


# ----------------------------------------------------
# CNN-based Encoder Networks
# ----------------------------------------------------
class StockCNNEncoder(nn.Module):
    def __init__(self, max_w, max_h, device):
        super(StockCNNEncoder, self).__init__()
        self.max_w = max_w
        self.max_h = max_h
        self.device = device
        
        def conv_output_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        self.kernel_size1 = min(8, max_w//4, max_h//4)  # Reduced from 8 
        self.stride1 = min(4, max_w//8, max_h//8)  # Reduced from 4
        self.kernel_size2 = min(4, max_w//8, max_h//8)  # Reduced from 4
        self.stride2 = min(2, max_w//16, max_h//16)  # Reduced from 2
        self.kernel_size3 = min(3, max_w//16, max_h//16)  # Reduced from 3
        self.stride3 = 1
        
        # Calculate intermediate output sizes
        h1 = conv_output_size(max_h, self.kernel_size1, self.stride1)
        w1 = conv_output_size(max_w, self.kernel_size1, self.stride1)
        
        h2 = conv_output_size(h1, self.kernel_size2, self.stride2) 
        w2 = conv_output_size(w1, self.kernel_size2, self.stride2)
        
        h3 = conv_output_size(h2, self.kernel_size3, self.stride3)
        w3 = conv_output_size(w2, self.kernel_size3, self.stride3)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=self.kernel_size1, stride=self.stride1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel_size2, stride=self.stride2) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=self.kernel_size3, stride=self.stride3)
        
        # Calculate flattened size for final linear layer
        self.flat_size = 64 * h3 * w3
        self.fc = nn.Linear(self.flat_size, 128)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class ProductEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, embed_dim=32, device=None):
        super(ProductEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.device = device

    def forward(self, prod_features):
        prod_features = prod_features.to(self.device)
        x = F.relu(self.fc1(prod_features))
        x = F.relu(self.fc2(x))
        return x

# ----------------------------------------------------
# Actor Network
# ----------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, num_stocks=100, num_products=20, max_w=100, max_h=100, device=None):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.num_stocks = num_stocks
        self.num_products = num_products
        self.max_w = max_w
        self.max_h = max_h

        self.stock_encoder = StockCNNEncoder(max_w, max_h, device).to(device)
        self.product_encoder = ProductEncoder(device=device).to(device)

        # Policy heads
        self.stock_head = nn.Linear(160, num_stocks).to(device)
        self.product_head = nn.Linear(160, num_products).to(device)

    def forward(self, obs):
        # Handle both single observations and batched observations
        if isinstance(obs, list):
            # Batch processing
            stocks = torch.stack([torch.tensor(np.array(o['stocks']), dtype=torch.float) for o in obs]).to(self.device)
            products_list = []
            for o in obs:
                prod_features = []
                for product in o['products']:
                    size = product['size']
                    quantity = product['quantity']
                    prod_features.append(np.concatenate((size, [quantity])))
                # Pad if needed
                pad_length = self.num_products - len(prod_features)
                if pad_length > 0:
                    prod_features += [[0, 0, 0]] * pad_length
                products_list.append(prod_features)
            products = torch.tensor(np.array(products_list), dtype=torch.float).to(self.device)
        else:
            # Single observation processing
            stocks = torch.tensor(np.array(obs['stocks']), dtype=torch.float).unsqueeze(0).to(self.device)
            # Process products
            products_list = []
            for product in obs['products']:
                size = product['size']
                quantity = product['quantity']
                products_list.append(np.concatenate((size, [quantity])))
            # Pad if needed
            pad_length = self.num_products - len(products_list)
            if pad_length > 0:
                products_list += [[0, 0, 0]] * pad_length
            products = torch.tensor(np.array([products_list]), dtype=torch.float).to(self.device)

        B = stocks.size(0)
        
        # Rest of processing remains the same
        stocks = stocks.unsqueeze(2)  # (B, num_stocks, 1, max_w, max_h)
        stocks = stocks.view(B * self.num_stocks, 1, self.max_h, self.max_w)
        stock_embeds = self.stock_encoder(stocks)  # (B*num_stocks, 128)
        stock_embeds = stock_embeds.view(B, self.num_stocks, 128)

        products = products.view(B * self.num_products, 3)
        product_embeds = self.product_encoder(products)  # (B*num_products, 32)
        product_embeds = product_embeds.view(B, self.num_products, 32)

        stock_summary = stock_embeds.mean(dim=1)     # (B, 128)
        product_summary = product_embeds.mean(dim=1)  # (B, 32)
        combined = torch.cat([stock_summary, product_summary], dim=-1)  # (B, 160)

        # For Actor Network
        if hasattr(self, 'stock_head'):
            stock_logits = self.stock_head(combined)       # (B, num_stocks)
            product_logits = self.product_head(combined)   # (B, num_products)
            return stock_logits, product_logits
        # For Critic Network
        else:
            value = self.value_head(combined)              # (B, 1)
            return value.squeeze(-1)  # (B,)


# ----------------------------------------------------
# Critic Network
# ----------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, num_stocks=100, num_products=20, max_w=100, max_h=100, device=None):
        super(CriticNetwork, self).__init__()
        self.device = device
        
        self.num_stocks = num_stocks
        self.num_products = num_products
        self.max_w = max_w
        self.max_h = max_h

        self.stock_encoder = StockCNNEncoder(max_w, max_h, device).to(device)
        self.product_encoder = ProductEncoder(device=device).to(device)

        # Value head
        self.value_head = nn.Linear(160, 1).to(device)


    def forward(self, obs):
        # Handle both single observations and batched observations
        if isinstance(obs, list):
            # Batch processing
            stocks = torch.stack([torch.tensor(np.array(o['stocks']), dtype=torch.float) for o in obs]).to(self.device)
            products_list = []
            for o in obs:
                prod_features = []
                for product in o['products']:
                    size = product['size']
                    quantity = product['quantity']
                    prod_features.append(np.concatenate((size, [quantity])))
                # Pad if needed
                pad_length = self.num_products - len(prod_features)
                if pad_length > 0:
                    prod_features += [[0, 0, 0]] * pad_length
                products_list.append(prod_features)
            products = torch.tensor(np.array(products_list), dtype=torch.float).to(self.device)
        else:
            # Single observation processing
            stocks = torch.tensor(np.array(obs['stocks']), dtype=torch.float).unsqueeze(0).to(self.device)
            # Process products
            products_list = []
            for product in obs['products']:
                size = product['size']
                quantity = product['quantity']
                products_list.append(np.concatenate((size, [quantity])))
            # Pad if needed
            pad_length = self.num_products - len(products_list)
            if pad_length > 0:
                products_list += [[0, 0, 0]] * pad_length
            products = torch.tensor(np.array([products_list]), dtype=torch.float).to(self.device)

        B = stocks.size(0)
        
        # Rest of processing remains the same
        stocks = stocks.unsqueeze(2)  # (B, num_stocks, 1, max_w, max_h)
        stocks = stocks.view(B * self.num_stocks, 1, self.max_h, self.max_w)
        stock_embeds = self.stock_encoder(stocks)  # (B*num_stocks, 128)
        stock_embeds = stock_embeds.view(B, self.num_stocks, 128)

        products = products.view(B * self.num_products, 3)
        product_embeds = self.product_encoder(products)  # (B*num_products, 32)
        product_embeds = product_embeds.view(B, self.num_products, 32)

        stock_summary = stock_embeds.mean(dim=1)     # (B, 128)
        product_summary = product_embeds.mean(dim=1)  # (B, 32)
        combined = torch.cat([stock_summary, product_summary], dim=-1)  # (B, 160)

        # For Actor Network
        if hasattr(self, 'stock_head'):
            stock_logits = self.stock_head(combined)       # (B, num_stocks)
            product_logits = self.product_head(combined)   # (B, num_products)
            return stock_logits, product_logits
        # For Critic Network
        else:
            value = self.value_head(combined)              # (B, 1)
            return value.squeeze(-1)  # (B,)
    
    
class PPO:
    def __init__(self, env, **hyperparameters):
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        
        observation, _ = env.reset(seed=None)
        self.num_stocks = len(observation["stocks"])
        self.max_h, self.max_w = observation["stocks"][0].shape
        self.num_products = env.unwrapped.max_product_type
        self.min_h = env.unwrapped.min_h
        self.min_w = env.unwrapped.min_w

        # Initialize actor and critic networks
        self.actor = ActorNetwork(
            num_stocks=self.num_stocks, 
            num_products=self.num_products,
            max_w=self.max_w,
            max_h=self.max_h,
            device=self.device
        ).to(self.device)
        
        self.critic = CriticNetwork(
            num_stocks=self.num_stocks,
            num_products=self.num_products, 
            max_w=self.max_w,
            max_h=self.max_h,
            device=self.device
        ).to(self.device)
        
        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'lr': 0,
        }
        
        self.max_product_area = self.min_h * self.min_w
    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()                     # ALG STEP 3
            
            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Normalizing advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            step = len(batch_obs)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger['lr'] = new_lr

                # Mini-batch Update
                np.random.shuffle(inds) # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = [batch_obs[i] for i in idx]  # Changed indexing for list
                    mini_acts = [batch_acts[i] for i in idx]  # Changed indexing for list
                    mini_log_prob = batch_log_probs[idx].to(self.device)
                    mini_advantage = A_k[idx].to(self.device)
                    mini_rtgs = batch_rtgs[idx].to(self.device)

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy Regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss                    
                    
                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break # if kl aboves threshold
            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                print("Saving actor and critic models...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                actor_path = os.path.join(current_dir, "ppo_actor.pth")
                critic_path = os.path.join(current_dir, "ppo_critic.pth")
                torch.save(self.actor.state_dict(), actor_path)
                torch.save(self.critic.state_dict(), critic_path)


    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float).to(self.device)


    def rollout(self):
       
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            # Reset the environment. Note that obs is short for observation. Set set to random seed 
            obs, _ = self.env.reset(seed=None)
            # Initially, the game is not done
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render:
                    self.env.render()
                # Track done flag of the current state
                ep_dones.append(done)

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.    
                action, log_prob, product_index, new_stock = self.get_action(obs)
        
                val = self.critic(obs)

                obs, rew, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated or ep_t == self.max_timesteps_per_episode - 1 or t == self.timesteps_per_batch
                rew = self.get_reward(obs, action, product_index, info, done, new_stock)
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    # Uncomment to print episode results
                    # if terminated:
                    #     print(f"Episode finished at timestep {ep_t}")
                    # else:
                    #     print(f"Episode truncated at timestep {ep_t}")
                    # print(info)
                    break

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals,batch_dones

    def get_action(self, obs):
        # Extract observation components
        stocks_np = obs['stocks']  # shape (num_stocks, max_w, max_h)
        products_np = obs['products']  # shape (num_products, 3)
        
        # Extract numerical data from products_np
        # Extract product features and quantities
        products_list = []
        for product in products_np:
            size = product['size']
            quantity = product['quantity']
            product_features = np.concatenate((size, [quantity]))
            products_list.append(product_features)

        # Calculate padding length
        pad_length = self.num_products - len(products_list)

        # Pad both arrays if needed
        if pad_length > 0:
            products_list += [[0, 0, 0]] * pad_length

        # Convert to numpy arrays
        products_array = np.array(products_list)  # Shape: (num_products, 3)
        
        # Query the actor network for a mean action
        stock_logits, product_logits = self.actor(obs)
        
        # Mask out products whose quantity is 0
        for i, product in enumerate(products_array):
            if product[2] == 0:
                product_logits[0][i] = -float('inf')

        # Create distributions
        # Sample an action from the distribution
        product_dist = Categorical(logits=product_logits)
        product_action = product_dist.sample()
        products_size = [products_array[product_action.item()][0], products_array[product_action.item()][1]]

        # Mask out stocks where product won't fit
        for i, stock in enumerate(stocks_np):
            act = greedy(obs['stocks'], i, products_size)
            if act['stock_idx'] == -1:
                stock_logits[0][i] = -float('inf')

        stock_dist = Categorical(logits=stock_logits)
        stock_action = stock_dist.sample()

        # Calculate the log probability for that action
        log_prob = stock_dist.log_prob(stock_action) + product_dist.log_prob(product_action)

        # Move action results back to CPU for numpy operations
        stock_action = stock_action.cpu()
        product_action = product_action.cpu()
        log_prob = log_prob.cpu()

        # Product size [w, h]
        action = greedy(obs['stocks'], stock_action.item(), products_size)
        
        is_new_stock = np.all(stocks_np[stock_action.item()] < 0)

        # Return the sampled action and the log probability of that action in our distribution
        return action, log_prob.detach(), product_action.item(), is_new_stock

    def evaluate(self, batch_obs, batch_acts):
        # Get critic's value prediction
        V = self.critic(batch_obs)

        # Get actor's action distributions
        stock_logits, product_logits = self.actor(batch_obs)
        
        # Create distributions
        stock_dist = Categorical(logits=stock_logits)
        product_dist = Categorical(logits=product_logits)

        # Extract stock indices from batch_acts and handle invalid indices
        batch_stock_indices = []
        batch_product_indices = []
        
        for obs, act in zip(batch_obs, batch_acts):
            # Handle invalid stock indices (-1) by replacing with 0
            # This is a temporary fix - the action will have very low probability
            stock_idx = act['stock_idx']
            if stock_idx < 0:
                stock_idx = 0  # Replace invalid index with valid one
            elif stock_idx >= self.num_stocks:
                stock_idx = self.num_stocks - 1  # Clip to valid range
            batch_stock_indices.append(stock_idx)
            
            # Find product index by matching sizes
            target_size = tuple(act['size'])
            found = False
            for i, product in enumerate(obs['products']):
                if tuple(product['size']) == target_size:
                    batch_product_indices.append(i)
                    found = True
                    break
            if not found:  # If no match found, use first product index
                batch_product_indices.append(0)

        # Convert to tensors and move to device
        batch_stock_indices = torch.tensor(batch_stock_indices).to(self.device)
        batch_product_indices = torch.tensor(batch_product_indices).to(self.device)

        # Calculate log probabilities
        stock_log_probs = stock_dist.log_prob(batch_stock_indices)
        product_log_probs = product_dist.log_prob(batch_product_indices)
        log_probs = stock_log_probs + product_log_probs

        # Calculate entropy
        entropy = stock_dist.entropy() + product_dist.entropy()

        return V, log_probs, entropy


    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = 0.98                                 # Lambda Parameter for GAE 
        self.num_minibatches = 6                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0.01                            # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02                           # KL Divergence threshold
        self.max_grad_norm = 0.5                        # Gradient Clipping threshold


        # Miscellaneous parameters
        self.render = False                             # If we should render during rollout
        self.save_freq = 10                             # How often we save in number of iterations
        self.deterministic = False                      # If we're testing, don't sample actions
        self.seed = None								# Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
        
        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        # Calculate logging values. 
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.cpu().float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        
        # Store log in CSV file
        log_file = "training_log.csv"
        headers = ['Iteration', 'Avg_Episode_Length', 'Avg_Episode_Reward', 'Avg_Loss', 'Timesteps', 'Delta_t', 'Learning_Rate']
        log_data = [i_so_far, avg_ep_lens, avg_ep_rews, avg_actor_loss, t_so_far, delta_t, lr]
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(log_data)

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    
    def get_reward(self, obs, action, product_idx, info, done, new_stock):
        # Remove invalid action penalty (since invalid actions are masked)

        # Calculate placement efficiency
        product_area = action['size'][0] * action['size'][1]
        # Set this to the actual maximum product area
        area_ratio = product_area / self.max_product_area

        if new_stock:
            # Penalize starting new stock moderately
            base_reward = -10.0
        else:
            # Reward using existing stock
            base_reward = 10.0
        
        # Scale reward by area ratio
        placement_reward = base_reward + area_ratio * 10.0  # Adjusted scaling
        
        if done:
            # Bonus for completion with emphasis on efficiency
            efficiency_bonus = (1.0 - info['trim_loss']) * 200.0  # Adjusted scaling
            placement_reward += efficiency_bonus
        # print(f"Reward: {placement_reward}")
        return placement_reward

    
def greedy(stocks, stock_idx, prod_size):
    if prod_size == [0, 0]:
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    stock = stocks[stock_idx]
    stock_w, stock_h = _get_stock_size_(stock)
    prod_w, prod_h = prod_size

    if stock_w >= prod_w and stock_h >= prod_h:
        for x in range(stock_w - prod_w + 1):
            find = 0
            for y in range(stock_h - prod_h + 1):
                if stock[x][y] == -1:
                    if find == 0:
                        if _can_place_(stock, (x, y), prod_size):
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}
                        find = 1
                else:
                    if find == 1:
                        find = 0

    if stock_w >= prod_h and stock_h >= prod_w:
        for x in range(stock_w - prod_h + 1):
            find = 0
            for y in range(stock_h - prod_w + 1):
                if stock[x][y] == -1:
                    if find == 0:
                        if _can_place_(stock, (x, y), prod_size[::-1]):
                            return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (x, y)}
                        find = 1
                else:
                    if find == 1:
                        find = 0

    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

def _get_stock_size_(stock):
    stock_w = np.sum(np.any(stock != -2, axis=1))
    stock_h = np.sum(np.any(stock != -2, axis=0))

    return stock_w, stock_h

def _can_place_(stock, position, prod_size):
    pos_x, pos_y = position
    prod_w, prod_h = prod_size

    return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename

    args = parser.parse_args()

    return args

def main(args):
    # NOTE: Here's where you can set hyperparameters for PPO.
    # For default gym_cutting_stock (num_stocks=100, max_product_type=25, max_product_per_type=20), 
    # training can take up to 8GB of RAM with timesteps_per_batch=100, so if you're running out 
    # of memory, consider reducing these values.
    # For example: gym_cutting_stock with num_stocks=16, max_product_type=5, max_product_per_type=10, 
    # use less RAM while training, so you can set timestep_per_batch=300 and 
    # max_timesteps_per_episode=60. 
    hyperparameters = {
                'timesteps_per_batch': 100, 
                'max_timesteps_per_episode': 100, 
                'gamma': 0.99, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2,
                'render': True,
                'render_every_i': 10
              }

    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0", 
        # min_w=50,
        # min_h=50,
        # max_w=100,
        # max_h=100,
        # num_stocks=100,
        # max_product_type=25,
        # max_product_per_type=20,
        # render_mode='human',
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a model for PPO with device
    model = PPO(env=env, **hyperparameters)

    # Try loading existing models with proper device mapping
    if args.actor_model != '' and args.critic_model != '':
        print(f"Loading in {args.actor_model} and {args.critic_model}...", flush=True)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        actor_path = os.path.join(current_dir, args.actor_model)
        critic_path = os.path.join(current_dir, args.critic_model)
        model.actor.load_state_dict(
            torch.load(actor_path, map_location=device)
        )
        model.critic.load_state_dict(
            torch.load(critic_path, map_location=device)
        )
        print(f"Successfully loaded.", flush=True)
    elif args.actor_model != '' or args.critic_model != '':
        print(f"Error: Either specify both actor/critic models or none at all.")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model
    model.learn(total_timesteps=200_000_000)

if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)