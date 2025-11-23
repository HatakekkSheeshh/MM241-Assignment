from policy import Policy
import numpy as np


class Policy2352649_2352732_2353055_2352576_2352708(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.placements = []  
        self.last_printed_episode = None  
        
    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._get_action_greedy(observation, info)
        elif self.policy_id == 2:
            return self._get_action_brute(observation, info)
    #grêdy
    def _get_action_greedy(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    pos_x, pos_y = self._find_adjacent_position(stock, prod_size, stock_w, stock_h)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                    prod_size = prod_size[::-1]
                    prod_w, prod_h = prod_size
                    pos_x, pos_y = self._find_adjacent_position(stock, prod_size, stock_w, stock_h)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    #du hoc duc 
    def _get_action_brute(self, observation, info):
        SMALL_PRODUCT_THRESHOLD = 5

        small_products = [prod for prod in observation["products"] if prod["quantity"] > 0 and max(prod["size"]) <= SMALL_PRODUCT_THRESHOLD]
        large_products = [prod for prod in observation["products"] if prod["quantity"] > 0 and max(prod["size"]) > SMALL_PRODUCT_THRESHOLD]

        sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1]))
        for prod in small_products:
            prod_size = prod["size"]
            for rotation in [prod_size, prod_size[::-1]]:
                prod_w, prod_h = rotation
                for idx, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    if stock_w >= prod_w and stock_h >= prod_h:
                        if self._can_place_(stock, (0, 0), rotation):
                            return {"stock_idx": idx, "size": rotation, "position": (0, 0)}
                        step_size = max(1, min(prod_w, prod_h) // 2)
                        possible_positions = self._generate_candidate_positions(stock_w, stock_h, prod_w, prod_h, step_size)
                        for x, y in possible_positions:
                            if self._can_place_(stock, (x, y), rotation):
                                return {"stock_idx": idx, "size": rotation, "position": (x, y)}

        list_prods = sorted(
            large_products,
            key=lambda p: (-max(p["size"]), -p["quantity"]),
        )

        best_action = None
        best_waste = float("inf")

        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue
            prod_size = prod["size"]
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                for rotation in [prod_size, prod_size[::-1]]:
                    prod_w, prod_h = rotation
                    if stock_w >= prod_w and stock_h >= prod_h:
                        possible_positions = self._generate_candidate_positions(stock_w, stock_h, prod_w, prod_h, i)
                        for x, y in possible_positions:
                            if self._can_place_(stock, (x, y), rotation):
                                waste = ((stock_w * stock_h) - (prod_w * prod_h) - self._calculate_used_area_(stock))
                                if waste < best_waste:
                                    best_waste = waste
                                    best_action = {"stock_idx": i, "size": rotation, "position": (x, y),}

        if best_action:
            self.placements.append(best_action)

        if info.get("episode") != self.last_printed_episode:
            if self.last_printed_episode is not None:
                print(f"Episode {self.last_printed_episode} results:", self.placements)
            self.placements = []
            self.last_printed_episode = info.get("episode")

        return best_action or {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _generate_candidate_positions(self, stock_w, stock_h, prod_w, prod_h, stock_idx):
        step_x = max(1, min(prod_w, prod_h) // 4)  
        step_y = max(1, min(prod_w, prod_h) // 4) 
        return [(x, y) for x in range(0, stock_w - prod_w + 1, step_x) for y in range(0, stock_h - prod_h + 1, step_y)]

    def _find_adjacent_position(self, stock, prod_size, stock_w, stock_h):
        prod_w, prod_h = prod_size

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    if self._is_adjacent(stock, x, y, prod_w, prod_h):
                        return x, y

        return None, None

    def _is_adjacent(self, stock, x, y, prod_w, prod_h):
        adjacent_positions = [ (x - 1, y), (x + prod_w, y), (x, y - 1), (x, y + prod_h) ]
        for adj_x, adj_y in adjacent_positions:
            if 0 <= adj_x < stock.shape[0] and 0 <= adj_y < stock.shape[1]:
                if stock[adj_x, adj_y] != -1:
                    return True

        return False

    def _calculate_used_area_(self, stock):
        return int(np.sum(stock != -1))
