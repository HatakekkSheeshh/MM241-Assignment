from policy import Policy
import numpy as np

class FFDPoicy(Policy):
    def __init__(self):
        self.patterns = []

    def get_action(self, observation, info):
        # Sort products by area in descending order
        products = sorted(
            observation["products"],
            key=lambda prod: -prod["size"][0] * prod["size"][1]
        )

        # Iterate through each product and try to place it in the first stock it fits
        for product in products:
            if product["quantity"] > 0:
                product_size = product["size"]
                rotated_size = product_size[::-1]

                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for size in [product_size, rotated_size]:
                        prod_w, prod_h = size

                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos = self._find_placement(stock, prod_w, prod_h)
                            if pos:
                                return {"stock_idx": stock_idx, "size": size, "position": pos}

        # No valid placement found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_placement(self, stock, prod_w, prod_h):
        stock_w, stock_h = self._get_stock_size_(stock)
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    return (x, y)
        return None

    def _get_stock_size_(self, stock):
        return stock.shape[0], stock.shape[1]

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        return np.all(stock[x:x+w, y:y+h] == -1)
    
class BranchAndBoundPolicy(Policy):
    def __init__(self):
        # Initialize necessary attributes for the Branch and Bound algorithm
        self.patterns = []  # List of stock cutting patterns
        self.branch_stack = []  # Stack to store branching states
        self.best_solution = None  # Best solution found so far
        self.placements = []  # List of successful placements
        self.max_depth = 10  # Maximum depth for recursion/backtracking

    def get_action(self, observation, info):
        # Retrieve the list of products and stocks from the observation
        products = observation["products"]
        stocks = observation["stocks"]

        # Initialize cutting patterns for stocks if not already initialized
        if not self.patterns:
            self._initialize_patterns(stocks)

        # Attempt to find the next placement using the backtracking method
        action = self._backtrack_placement(stocks, products, 0)
        if action:
            # If a valid action is found, append it to placements and return
            self.placements.append(action)
            return action

        # Return a default invalid placement if no action is found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _initialize_patterns(self, stocks):
        # Initialize cutting patterns for each stock
        for stock_idx, stock in enumerate(stocks):
            self.patterns.append({
                "stock_idx": stock_idx,
                "size": self._get_stock_size_(stock),  # Get stock dimensions
                "position": [],  # Positions of products placed in this stock
            })

    def _backtrack_placement(self, stocks, products, depth):
        # Terminate recursion if the maximum depth is reached or all products are placed
        if depth >= self.max_depth or not any(p["quantity"] > 0 for p in products):
            return None

        # Iterate through each stock to find a valid placement
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)  # Get stock dimensions

            # Iterate through all products to find one that can fit in the stock
            for prod_idx, product in enumerate(products):
                if product["quantity"] == 0:
                    continue  # Skip products with no remaining quantity

                # Try both orientations (original and rotated) of the product
                for orientation in [(product["size"], False), (product["size"][::-1], True)]:
                    prod_size, _ = orientation
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue  # Skip if product doesn't fit in the stock

                    # Check all possible positions in the stock for placement
                    x = 0
                    while x <= stock_w - prod_w:
                        y = 0
                        while y <= stock_h - prod_h:
                            if self._can_place_(stock, (x, y), prod_size):
                                # Create copies of stocks and products for branching
                                new_stocks = [np.copy(s) if i == stock_idx else s for i, s in enumerate(stocks)]
                                new_products = [dict(p) for p in products]

                                # Mark the area occupied by the product in the stock
                                new_stocks[stock_idx][x:x+prod_w, y:y+prod_h] = prod_idx
                                new_products[prod_idx]["quantity"] -= 1  # Decrement product quantity

                                # Return the placement details
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y),
                                    "depth": depth
                                }
                            y += 1
                        x += 1

                    # Recursively attempt placement in deeper levels
                    sub_placement = self._backtrack_placement(stocks, products, depth + 1)
                    if sub_placement:
                        return sub_placement

        # Return None if no valid placement is found
        return None


class Policy2352582_2352594_2352517(Policy):
    def __init__(self, policy_id=1):
        # Validate policy ID and initialize the corresponding policy
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = FFDPoicy() #Use FFD algoritm
        elif policy_id == 2:
            self.policy = BranchAndBoundPolicy()  # Use the Branch and Bound algorithm

    def get_action(self, observation, info):
        # Delegate the get_action call to the selected policy
        return self.policy.get_action(observation, info)





