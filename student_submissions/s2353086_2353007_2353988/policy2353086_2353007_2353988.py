from policy import Policy
import numpy as np

class Policy2353086_2353007_2353988(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy_get_action = self.smallest_stock_get_action
        elif policy_id == 2:
            self.policy_get_action = self.largest_stock_get_action

    def get_action(self, observation, info):
        """
        Unified method to call the appropriate policy's action.
        """
        return self.policy_get_action(observation, info)

    def smallest_stock_get_action(self, observation, info):
        """
        Policy to place products in the smallest stock that can accommodate them.
        """
        list_prods = observation["products"]

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Identify the smallest valid stock by area that can fit the product
                sorted_stocks = sorted(
                    enumerate(observation["stocks"]),
                    key=lambda x: np.sum(np.any(x[1] != -2, axis=1)) *
                                  np.sum(np.any(x[1] != -2, axis=0))
                )

                for stock_idx, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        # Try to place the product in the stock with the best fit
                        best_position = None
                        min_fit_difference = float('inf')

                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    # Calculate fit difference
                                    fit_difference = (stock_w * stock_h) - (prod_w * prod_h)

                                    if fit_difference < min_fit_difference:
                                        min_fit_difference = fit_difference
                                        best_position = (x, y)

                        if best_position:
                            pos_x, pos_y = best_position
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        # If no valid placement is found, return a default action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def largest_stock_get_action(self, observation, info):
        """
        Policy to place products in the largest stock that can accommodate them.
        """
        list_prods = observation["products"]

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Identify the largest valid stock by area that can fit the product
                sorted_stocks = sorted(
                    enumerate(observation["stocks"]),
                    key=lambda x: np.sum(np.any(x[1] != -2, axis=1)) *
                                  np.sum(np.any(x[1] != -2, axis=0)),
                    reverse=True
                )

                for stock_idx, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        # Try to place the product in the stock with the least wasted space
                        best_position = None
                        min_wasted_space = float('inf')

                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    # Calculate wasted space
                                    wasted_space = (stock_w * stock_h) - (prod_w * prod_h)

                                    if wasted_space < min_wasted_space:
                                        min_wasted_space = wasted_space
                                        best_position = (x, y)

                        if best_position:
                            pos_x, pos_y = best_position
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        # If no valid placement is found, return a default action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
