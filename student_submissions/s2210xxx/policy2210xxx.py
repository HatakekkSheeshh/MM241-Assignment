import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from policy import Policy

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        logits = self.network(x)
        # Use log_softmax instead of softmax to avoid numerical issues
        return torch.nn.functional.log_softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output is the state value V(s)
        )
    
    def forward(self, x):
        return self.network(x)

class Policy2210xxx(Policy):
    def __init__(self):
        self.state_dim = 200
        self.action_dim = 100
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # Add new parameters
        self.value_coef = 0.5  # Coefficient for value loss
        self.entropy_coef = 0.01  # Coefficient for entropy regularization
        
        # Initialize actor (policy) and critic networks
        self.actor = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        # Initialize optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Add list to store state values
        self.values = []
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def _encode_state(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Normalize features
        stocks_features = []
        total_space = 0
        used_space = 0
        
        for stock in stocks:
            stock_array = np.array(stock)
            available = (stock_array == -1).sum()
            used = (stock_array >= 0).sum()
            total_space += available + used
            used_space += used
            stocks_features.extend([available/100.0, used/100.0])  # Normalize values
            
        # Add overall usage ratio
        stocks_features.append(used_space/max(1, total_space))
        
        # Normalize product information
        products_features = []
        total_demand = 0
        for prod in products:
            size = prod["size"]
            quantity = prod["quantity"]
            total_demand += quantity * size[0] * size[1]
            products_features.extend([size[0]/10.0, size[1]/10.0, quantity/10.0])  # Normalize values
            
        # Padding features
        stocks_features = np.array(stocks_features, dtype=np.float32)
        products_features = np.array(products_features, dtype=np.float32)
        
        stocks_features = np.pad(stocks_features, 
                               (0, max(0, 100 - len(stocks_features))), 
                               mode='constant')[:100]
        
        products_features = np.pad(products_features, 
                                 (0, max(0, 100 - len(products_features))), 
                                 mode='constant')[:100]
        
        state = np.concatenate([stocks_features, products_features])
        return torch.FloatTensor(state).to(self.device)

    def get_action(self, observation, info):
        state = self._encode_state(observation)
        
        # Get value prediction from critic
        with torch.no_grad():
            state_value = self.critic(state)
            self.values.append(state_value)
        
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                log_probs = self.actor(state)
                probs = torch.exp(log_probs)
                action_idx = torch.multinomial(probs, 1).item()
        
        # Store log probability for training
        log_prob = self.actor(state)[action_idx]
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action_idx)
        
        return self._decode_action(action_idx, observation)

    def update_policy(self):
        if len(self.rewards) == 0:
            return
            
        # Tính returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device, requires_grad=True)  # Add requires_grad=True

        # Convert lists to tensors
        values = torch.cat(self.values)
        log_probs = torch.stack(self.log_probs)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate actor (policy) loss
        policy_loss = []
        entropy = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage.detach())  # Detach advantage
            probs = torch.exp(log_prob)
            entropy.append(-(probs * log_prob).sum())
        
        policy_loss = torch.stack(policy_loss).sum()
        entropy_loss = -self.entropy_coef * torch.stack(entropy).sum()
        actor_loss = policy_loss + entropy_loss
        
        # Calculate critic (value) loss
        values = values.view(-1)  # Flatten values
        returns = returns.view(-1)  # Flatten returns
        value_loss = self.value_coef * F.mse_loss(values, returns)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # Thêm retain_graph=True
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Decrease epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Reset memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def _decode_action(self, action_idx, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Filter products that still need to be cut
        valid_products = [(i, p) for i, p in enumerate(products) if p["quantity"] > 0]
        if not valid_products:
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
        
        # Sort products by area in descending order
        valid_products.sort(key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
        
        # Find optimal stock for each product
        for prod_idx, prod in valid_products:
            prod_size = prod["size"]
            best_stock = None
            best_position = None
            min_waste = float('inf')
            
            # Iterate over stocks that are unused or partially used
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size
                
                # Skip stocks that are too small
                if stock_w < prod_w or stock_h < prod_h:
                    continue
                    
                # Find optimal position in the current stock
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            # Calculate space waste
                            waste = self._calculate_waste(stock, (x, y), prod_size)
                            if waste < min_waste:
                                min_waste = waste
                                best_stock = stock_idx
                                best_position = (x, y)
                
                # Stop searching if a good position is found
                if best_position is not None and min_waste < stock_w * stock_h * 0.1:  # Waste threshold 10%
                    break
                    
            if best_position is not None:
                return {
                    "stock_idx": best_stock,
                    "size": prod_size,
                    "position": best_position
                }
        
        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
    def _calculate_waste(self, stock, position, size):
        """Calculate the area of waste when placing a product"""
        x, y = position
        w, h = size
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        
        # Calculate empty area around the placement
        empty_area = 0
        for i in range(max(0, x-1), min(stock_w, x+w+1)):
            for j in range(max(0, y-1), min(stock_h, y+h+1)):
                if stock[i,j] == -1:
                    empty_area += 1
                    
        return empty_area - w*h

