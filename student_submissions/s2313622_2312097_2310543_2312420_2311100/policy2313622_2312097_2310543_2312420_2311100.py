from policy import Policy
from .ppo_optimized import PPO, ActorNetwork, CriticNetwork
import torch
import numpy as np
from torch.distributions import Categorical
import os
import torch.nn as nn
import torch.optim as optim


class Policy2313622_2312097_2310543_2312420_2311100(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2 , 3], "Policy ID must be 1 or 2 or 3"

        # Student code here
        if policy_id == 1:
            
            #-------------------------------------------------------------------------------#
            # NOTE: The following policy was trained for the default gym_cutting_stock      #
            # environment parameter:                                                        #
            # min_w=50,                                                                     #
            # min_h=50,                                                                     #
            # max_w=100,                                                                    #
            # max_h=100,                                                                    #
            # num_stocks=100,                                                               #
            # max_product_type=25,                                                          #
            # max_product_per_type=20,                                                      #
            # If you use different parameters, you have to retrain the policy for your      #
            # environment, using the provided ppo_optimized.py script; otherwise, the       #
            # policy will not work correctly.                                               #
            #-------------------------------------------------------------------------------#
            
            self.num_stocks = 100   # Change this value if you use a different environment
            self.max_product_type = 25  # Change this value if you use a different environment

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = ActorNetwork(num_stocks=self.num_stocks, num_products=self.max_product_type, device=device).to(device)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            policy_path = os.path.join(current_dir, "ppo_actor.pth")
            policy.load_state_dict(
                torch.load(
                    policy_path,
                    map_location=device
                )
            )
            self.id = policy_id
            self.policy = policy
        elif policy_id == 2:
            self.id = policy_id
            self.policy = self.Heuristic_RL_Policy()
            
        elif policy_id == 3:
            self.id = policy_id
            self.policy = self.HeuristicPolicy()
            
        
            

    def get_action(self, observation, info):
        # Student code here
        
        if self.id == 1:
            obs = observation
            
            # Extract observation components
            stocks_np = obs['stocks']  # shape (num_stocks, 100, 100)
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
            pad_length = self.max_product_type - len(products_list)

            # Pad both arrays if needed
            if pad_length > 0:
                products_list += [[0, 0, 0]] * pad_length

            # Convert to numpy arrays
            products_array = np.array(products_list)  # Shape: (num_products, 3)
            
            # Query the actor network for a mean action
            stock_logits, product_logits = self.policy(obs)
            
            # Mask out products whose quantity is 0
            for i, product in enumerate(products_array):
                if product[2] == 0:
                    product_logits[0][i] = -float('inf')

            # Sample an action from the distribution
            product_dist = Categorical(logits=product_logits)
            product_action = product_dist.sample()
            products_size = [products_array[product_action.item()][0], products_array[product_action.item()][1]]

            # Mask out stocks where product won't fit
            for i, stock in enumerate(stocks_np):
                act = self.greedy(obs['stocks'], i, products_size)
                if act['stock_idx'] == -1:
                    stock_logits[0][i] = -float('inf')

            stock_dist = Categorical(logits=stock_logits)
            stock_action = stock_dist.sample()
            
            # Move action results back to CPU for numpy operations
            stock_action = stock_action.cpu()
            product_action = product_action.cpu()

            # Product size [w, h]
            action = self.greedy(obs['stocks'], stock_action.item(), products_size)

            # Return the sampled action and the log probability of that action in our distribution
            return action

        elif self.id == 2 or self.id == 3:
            return self.policy.get_action(observation, info)
        
    # Student code here
    # You can add more functions if needed
    def greedy(self, stocks, stock_idx, prod_size):
        if prod_size == [0, 0]:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
        stock = stocks[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w >= prod_w and stock_h >= prod_h:
            for x in range(stock_w - prod_w + 1):
                find = 0
                for y in range(stock_h - prod_h + 1):
                    if stock[x][y] == -1:
                        if find == 0:
                            if self._can_place_(stock, (x, y), prod_size):
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
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (x, y)}
                            find = 1
                    else:
                        if find == 1:
                            find = 0

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    class HeuristicPolicy(Policy):
        def __init__(self):
            self.prod_indice = []
            self.sorted_stock = []
            self.solutions = []
            self.actions = []
            self.prods = []

        def get_action(self, observation, info):
            


            if info["filled_ratio"] == 0:
                self.reset(observation["products"], observation["stocks"])

            while self.actions or self.solutions or self.sorted_stock:

                if not self.actions:
                    self.new_action(observation["products"])

                if self.actions:
                    return self.actions.pop(0)
            
                if not self.solutions:
                    self.new_solution(observation["products"])

                if not self.solutions:
                    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        def reset(self, prods, stocks):
            self.prods = []
            for prod in prods:
                if self.prods:
                    invalid = 0
                    for self_prod in self.prods:
                        if (self_prod["size"][0] == prod["size"][0] and self_prod["size"][1] == prod["size"][1]) or (self_prod["size"][0] == prod["size"][1] and self_prod["size"][1] == prod["size"][0]):
                            self_prod["quantity"] += prod["quantity"]
                            invalid = 1
                            break
                    if invalid == 0:
                        self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
                else:
                    self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
            self.actions = []
            self.solutions = []
            self.prod_indice = sorted(range(len(self.prods)), key=lambda i: self.prods[i]["size"][0] * self.prods[i]["size"][1], reverse=True)
            self.sorted_stock = sorted([(self._get_stock_size_(stock), i) for i, stock in enumerate(stocks)], key=lambda x: x[0][0] * x[0][1], reverse=False)       

        def new_solution(self, prods):
            if not self.prod_indice:
                return

            self.prod_indice = [idx for idx in self.prod_indice if self.prods[idx]["quantity"] != 0 ]
            
            #while prods[self.prod_indice[0]]["quantity"] == 0:
            #    self.prod_indice.pop(0)

            prod_w, prod_h = self.prods[self.prod_indice[0]]["size"]
            quantity = self.prods[self.prod_indice[0]]["quantity"]

            # find smallest prod can be used
            max_stock = 0
            _max = -1

            for stock in self.sorted_stock:
                stock_w, stock_h = stock[0]
                if stock_w >= prod_w and stock_h >= prod_h:
                    if (stock_w // prod_w) * (stock_h // prod_h) > _max:
                        _max = (stock_w // prod_w) * (stock_h // prod_h)
                        max_stock = stock
                        if _max >= quantity:
                            break

                if stock_w >= prod_h and stock_h >= prod_w:
                    if (stock_w // prod_h) * (stock_h // prod_w) > _max:
                        _max = (stock_w // prod_h) * (stock_h // prod_w)
                        max_stock = stock
                        if _max >= quantity:
                            break

            if _max == -1:
                return
            self.solutions = [{"stock_idx": max_stock[1], "size": max_stock[0], "position": (0, 0)}]
            self.sorted_stock.remove(max_stock)

        def new_action(self, prods):
            # choose product
            if not self.solutions:
                return

            stock_idx = self.solutions[0]["stock_idx"]
            #print(self.solutions)
            #print(prods)
            #print("###")
            while self.solutions:
                solution = self.solutions[0]
                stock_w, stock_h = solution["size"]
                self.solutions.pop(0)
                #print(solution)
                # Try to find the first fit product
                for prod_idx in self.prod_indice:
                    prod = self.prods[prod_idx]
            
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        prod_w, prod_h = prod_size

                        if stock_w >= prod_w and stock_h >= prod_h:
                            if stock_w >= prod_h and stock_h >= prod_w:
                                if (stock_w // prod_w) * (stock_h // prod_h) >= (stock_w // prod_h) * (stock_h // prod_w):
                                    prod["quantity"] -= self.get_solution(solution, prod)
                                    #print(prod_idx)
                                    return
                                else:
                                    prod["size"] = prod["size"][::-1]
                                    prod["quantity"] -= self.get_solution(solution, prod)
                                    prod["size"] = prod["size"][::-1]
                                    #print(prod_idx)
                                    return
                            prod["quantity"] -= self.get_solution(solution, prod)
                            #print(prod_idx)
                            return
                        else:
                            if stock_w >= prod_h and stock_h >= prod_w:
                                prod["size"] = prod["size"][::-1]
                                prod["quantity"] -= self.get_solution(solution, prod)
                                prod["size"] = prod["size"][::-1]
                                #print(prod_idx)
                                return

        def get_solution(self, solution, prod):
            quantity = prod["quantity"]
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            stock_w, stock_h = solution["size"]
            stock_x, stock_y = solution["position"]
            stock_idx = solution["stock_idx"]

            cut_w = stock_w // prod_w
            cut_h = stock_h // prod_h

            if cut_w * cut_h > quantity:
                cut_h = quantity // cut_w
            if cut_h == 0:
                cut_w = quantity
                cut_h = 1

            for x in range(cut_w):
                for y in range(cut_h):
                    self.actions.append({"stock_idx": stock_idx, "size": prod_size, "position": (stock_x + x * prod_w, stock_y + y * prod_h)})

            prod_w *= cut_w
            prod_h *= cut_h

            # add solutions
            if (stock_w - prod_w) * stock_h < (stock_h - prod_h) * stock_w:
                if stock_w - prod_w != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [stock_w - prod_w, prod_h], "position": (stock_x + prod_w, stock_y)})
                if stock_h - prod_h != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [stock_w, stock_h - prod_h], "position": (stock_x, stock_y + prod_h)})
            else:
                if stock_w - prod_w != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [stock_w - prod_w, stock_h], "position": (stock_x + prod_w, stock_y)})
                if stock_h - prod_h != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [prod_w, stock_h - prod_h], "position": (stock_x, stock_y + prod_h)})

            self.solutions = sorted(self.solutions, key=lambda x: min(x["size"]), reverse=False)
            return cut_w * cut_h

    class Heuristic_RL_Policy(HeuristicPolicy):

        def train(self, observation, info, list_m, list_n, train):
            if info["filled_ratio"] == 0:
                self.prods = []
                for prod in observation["products"]:
                    if self.prods:
                        invalid = 0
                        for self_prod in self.prods:
                            if (self_prod["size"][0] == prod["size"][0] and self_prod["size"][1] == prod["size"][1]) or (self_prod["size"][0] == prod["size"][1] and self_prod["size"][1] == prod["size"][0]):
                                self_prod["quantity"] += prod["quantity"]
                                invalid = 1
                                break
                        if invalid == 0:
                            self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
                    else:
                        self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
                self.actions = []
                self.solutions = []
                self.prod_indice = sorted(range(len(self.prods)), key=lambda i: list_m[i], reverse=True)
                self.sorted_stock = [((int(stock[0][0]),int(stock[0][1])), int(stock[1])) for stock in observation["stocks"]]
                self.sorted_stock = sorted(self.sorted_stock, key=lambda x: list_n[x[1]], reverse=True)

            while self.actions or self.solutions or self.sorted_stock:

                if not self.actions:
                    self.new_action(observation["products"])

                if self.actions:
                    return self.actions.pop(0)
            
                if not self.solutions:
                    self.new_solution(observation["products"])

                if not self.solutions:
                    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        def reset(self, prods, stocks, _try = 20, _iteration = 10, _number_of_model = 10):
            self.prods = []
            for prod in prods:
                if self.prods:
                    invalid = 0
                    for self_prod in self.prods:
                        if (self_prod["size"][0] == prod["size"][0] and self_prod["size"][1] == prod["size"][1]) or (self_prod["size"][0] == prod["size"][1] and self_prod["size"][1] == prod["size"][0]):
                            self_prod["quantity"] += prod["quantity"]
                            invalid = 1
                            break
                    if invalid == 0:
                        self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
                else:
                    self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
            sorted_stock = [(((stock_size := self._get_stock_size_(stock))[0], stock_size[1]),i) for i, stock in enumerate(stocks)]
            
            # FIRST PREDICT
            env = Policy2313622_2312097_2310543_2312420_2311100.My_Environment(prods, stocks)
            policy = Policy2313622_2312097_2310543_2312420_2311100.HeuristicPolicy()

            env_prods, env_info = env.reset(prods, stocks)
            env_terminated = 0
            while env_terminated == 0:
                action = policy.get_action({"products": env_prods, "stocks": stocks}, env_info)
                env_prods, env_info, env_terminated = env.step(action)
            _mean = env_info["trim_loss"].copy()
            _origin_mean = env_info["trim_loss"].copy()

            list_model = []
            list_trim_loss = []
            for _ in range(_number_of_model):
                list_model.append(Policy2313622_2312097_2310543_2312420_2311100.Model(len(self.prods) + len(stocks), len(self.prods) + len(stocks)))
                list_trim_loss.append(1)          
            
            # BEGIN TRAINING ###################
            self.min_trim_loss = env_info["trim_loss"].copy()
            for _ in range(_try):
                for idx_model in range(_number_of_model):
                    for _ in range(_iteration):
                        list_m, list_n = list_model[idx_model].train(env, self.prods, sorted_stock)
                    
                        env_prods, env_info = env.reset(prods, stocks)
                        env_terminated = 0
                        while env_terminated == 0:
                            action = self.train({"products": env_prods, "stocks": sorted_stock}, env_info, list_m, list_n, 1)
                            env_prods, env_info, env_terminated = env.step(action)
                        #print(env_info)
                    
                        if (self.min_trim_loss > env_info["trim_loss"]):
                            self.min_trim_loss = env_info["trim_loss"].copy()
                            self.min_list_m = list_m.copy()
                            self.min_list_n = list_n.copy()

                        reward = (-env_info["trim_loss"] + _mean)
                        list_model[idx_model].update_parameters(reward)

                    list_m, list_n = list_model[idx_model].train(env, self.prods, sorted_stock)
                    
                    env_prods, env_info = env.reset(prods, stocks)
                    env_terminated = 0
                    while env_terminated == 0:
                        action = self.train({"products": env_prods, "stocks": sorted_stock}, env_info, list_m, list_n, 1)
                        env_prods, env_info, env_terminated = env.step(action)
                    #print(env_info)
                    list_model[idx_model].clear()
                    list_trim_loss[idx_model] = env_info["trim_loss"].copy()
                
                old_mean = sum(list_trim_loss) / len(list_trim_loss)
                good_ans = [trim_loss for trim_loss in list_trim_loss if trim_loss <= old_mean]
                new_mean = sum(good_ans) / len(good_ans)
                
                if (new_mean > _origin_mean):
                    mean = (_origin_mean + new_mean) / 2
                else:
                    mean = new_mean
                
                for idx_model in range(_number_of_model):
                    if list_trim_loss[idx_model] > old_mean:
                        list_model[idx_model].reset_model()
            # END TRAINING #############
            
            self.prods = []
            for prod in prods:
                if self.prods:
                    invalid = 0
                    for self_prod in self.prods:
                        if (self_prod["size"][0] == prod["size"][0] and self_prod["size"][1] == prod["size"][1]) or (self_prod["size"][0] == prod["size"][1] and self_prod["size"][1] == prod["size"][0]):
                            self_prod["quantity"] += prod["quantity"]
                            invalid = 1
                            break
                    if invalid == 0:
                        self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
                else:
                    self.prods.append({"size": (int(prod["size"][0]),int(prod["size"][1])), "quantity": int(prod["quantity"])})
            self.actions = []
            self.solutions = []
            if (self.min_trim_loss < _origin_mean):
                self.prod_indice = sorted(range(len(self.prods)), key=lambda i: self.min_list_m[i], reverse=True)
                self.sorted_stock = sorted(sorted_stock, key=lambda x: self.min_list_n[x[1]], reverse=True)
            else:
                self.prod_indice = sorted(range(len(self.prods)), key=lambda i: self.prods[i]["size"][0] * self.prods[i]["size"][1], reverse=True)
                self.sorted_stock = sorted(sorted_stock, key=lambda x: x[0][0] * x[0][1], reverse=False)    

    class My_Environment(Policy):

        def __init__(self, prods, stocks):
            self.prods = [{"size": (int(prod["size"][0]), int(prod["size"][1])), "quantity": int(prod["quantity"])} for prod in prods]
            self.stocks = [(self._get_stock_size_(stock)) for stock in stocks]
            self.info = {"filled_ratio": 0, "trim_loss": 0}
            self.used_stocks = []
            self.total = 0
        
        def reset(self, prods, stocks):
            self.prods = [{"size": (int(prod["size"][0]), int(prod["size"][1])), "quantity": int(prod["quantity"])} for prod in prods]
            #self.stocks = [(self._get_stock_size_(stock)) for stock in stocks]
            self.info = {"filled_ratio": 0, "trim_loss": 0}
            self.used_stocks = []
            self.total = 0
            return self.prods, self.info

        def step(self, action):
            if action["stock_idx"] == -1:
                return self.prods, self.info, -1

            # get value
            action_w, action_h = action["size"]
            action_idx = action["stock_idx"]

            # check new stock
            new_stock = 1
            if self.used_stocks:
                for used_stock in self.used_stocks:
                    if used_stock == action_idx:
                        new_stock = 00
            
            
            if new_stock == 1:
                self.used_stocks.append(action_idx)
                self.info["filled_ratio"] = len(self.used_stocks) / len(self.stocks)
            
            self.total += (action_w * action_h) / (self.stocks[action_idx][0] * self.stocks[action_idx][1])
            self.info["trim_loss"] = (len(self.used_stocks) - self.total) / len(self.used_stocks)

            terminated = 1
            for prod in self.prods:
                prod_w, prod_h = prod["size"]
                if ((prod_w == action_w and prod_h == action_h) or (prod_w == action_h and prod_h == action_w)) and prod["quantity"] > 0:
                    prod["quantity"] -= 1
                if prod["quantity"] != 0:
                    terminated = 0
            return self.prods, self.info, terminated


    class ActorCritic(nn.Module):
        def __init__(self, input_size, output_size, hidden_size=256):
            super(Policy2313622_2312097_2310543_2312420_2311100.ActorCritic, self).__init__()
            self.actor = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid()
            )
            self.critic = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)  # Value function estimation
            )

        def forward(self, x):
            action_probs = self.actor(x)
            value = self.critic(x)
            return action_probs, value

    class Model(HeuristicPolicy):

        def reset_weights(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        def reset_model(self):
            self.model.apply(Policy2313622_2312097_2310543_2312420_2311100.Model.reset_weights)

        def clear(self):
            self.probabilities_ = []
            self.value_ = []

        def __init__(self, input_size, output_size, hidden_size=128, lr=0.0001):
            self.model = Policy2313622_2312097_2310543_2312420_2311100.ActorCritic(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
            self.input_size = input_size
            self.output_size = output_size
            self.hidden_size = hidden_size
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.prediction_count = 0
            #self.gamma = 0.99
            self.clip_epsilon = 0.2
            self.c_v = 1
            self.beta = 0.01
            self.probabilities_ = []
            self.value_ = []

        def train(self, env__, prods, sorted_stocks):

            input_prods = [prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in prods]
            input_stocks = [stock[0][0]* stock[0][1] for stock in sorted_stocks]
            max_size = max(input_stocks) * 1.25
            tensor_prods = torch.tensor(input_prods, dtype=torch.float32) / max_size
            tensor_stocks = torch.tensor(input_stocks, dtype=torch.float32) / max_size
            input_tensor = torch.cat((tensor_prods, tensor_stocks), dim=0)
            input_tensor = input_tensor.unsqueeze(0)

            # forward
            action_probs, value = self.model(input_tensor)

            tensor_m = action_probs[:, :len(prods)]
            tensor_n = action_probs[:, len(prods):len(prods) + len(sorted_stocks)]
            list_m = tensor_m.squeeze(0).tolist()
            list_n = tensor_n.squeeze(0).tolist()

            self.probabilities_.append(action_probs)
            self.value_.append(value)

            return list_m, list_n

        def predict(self, prods, sorted_stocks):
            # builld input
            input_prods = [prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in prods]
            input_stocks = [stock[0][0]* stock[0][1] for stock in sorted_stocks]
            max_size = max(input_stocks) * 1.25
            tensor_prods = torch.tensor(input_prods, dtype=torch.float32) / max_size
            tensor_stocks = torch.tensor(input_stocks, dtype=torch.float32) / max_size
            input_tensor = torch.cat((tensor_prods, tensor_stocks), dim=0)
            input_tensor = input_tensor.unsqueeze(0)

            # forward
            action_probs, value = self.model(input_tensor)

            tensor_m = action_probs[:, :len(prods)]
            tensor_n = action_probs[:, len(prods):len(prods) + len(sorted_stocks)]
            list_m = tensor_m.squeeze(0).tolist()
            list_n = tensor_n.squeeze(0).tolist()

            return list_m, list_n

        def reset(self, prods, stocks):
            self.actions = []
            self.solutions = []
            self.sorted_stock = [(((stock_size := self._get_stock_size_(stock))[0], stock_size[1]),i) for i, stock in enumerate(stocks)]

            # builld input
            input_prods = [prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in prods]
            input_stocks = [stock[0][0]* stock[0][1] for stock in self.sorted_stock]
            max_size = max(input_stocks) * 1.25
            tensor_prods = torch.tensor(input_prods, dtype=torch.float32) / max_size
            tensor_stocks = torch.tensor(input_stocks, dtype=torch.float32) / max_size
            input_tensor = torch.cat((tensor_prods, tensor_stocks), dim=0)
            input_tensor = input_tensor.unsqueeze(0)

            # forward
            action_probs, value = self.model(input_tensor)

            tensor_m = action_probs[:, :len(prods)]
            tensor_n = action_probs[:, len(prods):len(prods) + len(self.sorted_stock)]
            list_m = tensor_m.squeeze(0).tolist()
            list_n = tensor_n.squeeze(0).tolist()

            # save
            self.probabilities_.append(action_probs)
            self.value_.append(value)

            self.prod_indice = sorted(range(len(prods)), key=lambda i: list_m[i], reverse=True)
            self.sorted_stock = sorted(self.sorted_stock, key=lambda x: list_n[x[1]], reverse=True)

        def update_parameters(self, reward):
            """
            Update parameters using PPO policy loss and value loss.
            """
            # Dummy reward and advantage for demonstration
            if (len(self.probabilities_) > 1 or len(self.value_) > 1):
                print("fail")
            
            action_probs = self.probabilities_.pop()
            value = self.value_.pop()
            
            #rewards = []
            #discounted_reward = 0
            #for reward in reversed(reward):
            #    discounted_reward = reward + self.gamma * discounted_reward
            #    rewards.insert(0, discounted_reward)
            #rewards = torch.tensor(rewards)
            rewards = torch.tensor(reward).unsqueeze(-1)
            #print(sum(rewards))
            advantage = rewards - value.detach()

            # PPO Loss Calculation
            old_probs = action_probs.detach()
            log_probs = torch.log(action_probs + 1e-10)
            ratio = torch.exp(log_probs - torch.log(old_probs + 1e-10))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage

            entropy = -(action_probs.detach() * torch.log(action_probs.detach() + 1e-10)).sum(dim=-1).mean()
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (rewards - value).pow(2).mean()

            # Combined loss
            loss = policy_loss + self.c_v * value_loss - self.beta * entropy
            #print(loss)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
