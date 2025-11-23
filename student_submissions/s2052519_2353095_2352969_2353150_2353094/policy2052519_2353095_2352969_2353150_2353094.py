import numpy as np

from policy import Policy
from scipy.optimize import linprog

class FirstCutPolicy:
    def get_action(self, observation, info):
        stocks = observation['stocks']
        products = observation['products']
        
        for product in products: # Needs optimization for runtime
            product_size = product['size']
            demand = product['quantity']
            
            for stock_idx, stock in enumerate(stocks):
                stock_height, stock_width = stock.shape
                product_height, product_width = product_size
                
                if product_height <= stock_height and product_width <= stock_width and demand > 0:
                    for i in range(stock_height - product_height + 1): # O(n^2) -> Needs optimization
                        for j in range(stock_width - product_width + 1):
                            subgrid = stock[i:i + product_height, j:j + product_width]
                            if np.all(subgrid == -1):
                                action = {
                                    "stock_idx": stock_idx,
                                    "size": (product_height, product_width),
                                    "position": (i, j),
                                }

                                product['quantity'] -= 1
                                return action

        return None

class ColumnGeneration(Policy):
    def __init__(self):
        super().__init__()
        self.patterns = []
        self.is_initialized = False
        self.num_products = 0

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        demand = np.array([product["quantity"] for product in products])
        sizes = [product["size"] for product in products]
        num_products = len(products)

        if not self.is_initialized or self.num_products != num_products:
            self._setup_initial_patterns(num_products, sizes, stocks)
            self.is_initialized = True
            self.num_products = num_products

        while True:
            coefficients = np.ones(len(self.patterns))
            constraint_matrix = np.array(self.patterns).T
            demand_constraints = demand

            result = linprog(
                c=coefficients,
                A_ub=-constraint_matrix,
                b_ub=-demand_constraints,
                bounds=(0, None),
                method="highs"
            )

            if result.status != 0:
                break 

            dual_prices = self._extract_duals(result, demand)
            if dual_prices is None:
                break

            new_pattern = self._find_new_pattern(dual_prices, sizes, stocks)
            if new_pattern is None or any((new_pattern == p).all() for p in self.patterns):
                break

            self.patterns.append(new_pattern)

        chosen_pattern = self._choose_pattern(demand)
        return self._convert_to_action(chosen_pattern, sizes, stocks)

    def _setup_initial_patterns(self, num_products, sizes, stocks):
        """Generate initial patterns by checking all products against available stocks."""
        self.patterns = []
        for stock in stocks:
            stock_dimensions = self._get_stock_size_(stock)
            for i, product_size in enumerate(sizes):
                if self._fits_in_stock(product_size, stock_dimensions):
                    pattern = np.zeros(num_products, dtype=int)
                    pattern[i] = 1
                    self.patterns.append(pattern)

        # Ensure unique patterns
        self.patterns = [np.array(p) for p in {tuple(p): p for p in self.patterns}.values()]

    def _extract_duals(self, linprog_result, demand):
        """Extract dual prices from the optimization result."""
        if hasattr(linprog_result, "ineqlin") and hasattr(linprog_result.ineqlin, "marginals"):
            return linprog_result.ineqlin.marginals
        return None

    def _find_new_pattern(self, dual_prices, sizes, stocks):
        """Generate a new pattern using a heuristic greedy approach."""
        best_pattern = None
        highest_value = -float("inf")

        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            dp_table = np.zeros((stock_h + 1, stock_w + 1))
            candidate_pattern = np.zeros(len(sizes), dtype=int)

            for i, size in enumerate(sizes):
                prod_w, prod_h = size
                if prod_w <= stock_w and prod_h <= stock_h and dual_prices[i] > 0:
                    for w in range(stock_w, prod_w - 1, -1):
                        for h in range(stock_h, prod_h - 1, -1):
                            dp_table[h][w] = max(
                                dp_table[h][w], dp_table[h - prod_h][w - prod_w] + dual_prices[i]
                            )

            current_w, current_h = stock_w, stock_h
            for i in range(len(sizes) - 1, -1, -1):
                prod_w, prod_h = sizes[i]
                if (
                    current_w >= prod_w
                    and current_h >= prod_h
                    and dp_table[current_h][current_w]
                    == dp_table[current_h - prod_h][current_w - prod_w] + dual_prices[i]
                ):
                    candidate_pattern[i] += 1
                    current_w -= prod_w
                    current_h -= prod_h

            value = np.dot(candidate_pattern, dual_prices) - 1
            if value > highest_value:
                highest_value = value
                best_pattern = candidate_pattern

        return best_pattern if highest_value > 0 else None

    def _choose_pattern(self, demand):
        """Choose the pattern that maximizes product demand coverage."""
        max_coverage = -1
        selected_pattern = None

        for pattern in self.patterns:
            coverage = np.sum(np.minimum(pattern, demand))
            if coverage > max_coverage:
                max_coverage = coverage
                selected_pattern = pattern

        return selected_pattern

    def _convert_to_action(self, pattern, sizes, stocks):
        """Convert the chosen pattern into an actionable cutting plan."""
        for product_idx, count in enumerate(pattern):
            if count > 0:
                product_size = sizes[product_idx]
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    if stock_w >= product_size[0] and stock_h >= product_size[1]:
                        placement = self._find_placement(stock, product_size)
                        if placement:
                            return {"stock_idx": stock_idx, "size": product_size, "position": placement}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_placement(self, stock, product_size):
        """Find a valid placement for the product in the stock."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product_size

        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), product_size):
                    return (x, y)

        return None

    def _fits_in_stock(self, product_size, stock_size):
        """Check if a product can fit in a stock."""
        prod_w, prod_h = product_size
        stock_w, stock_h = stock_size
        return prod_w <= stock_w and prod_h <= stock_h
    

class Policy2052519_2353095_2352969_2353150_2353094(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy = None
        if policy_id == 1:
            self.policy = FirstCutPolicy()
        elif policy_id == 2:
            self.policy = ColumnGeneration()

    def get_action(self, observation, info):
        # Student code here
        if self.policy is not None:
            return self.policy.get_action(observation, info)
        else:
            raise NotImplementedError("Policy is not implemented yet.")

    # Student code here
    # You can add more functions if needed