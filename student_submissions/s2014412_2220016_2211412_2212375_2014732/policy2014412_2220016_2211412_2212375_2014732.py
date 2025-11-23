from policy import Policy
import numpy as np
import random


class Policy2014412_2220016_2211412_2212375_2014732(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = 1
        elif policy_id == 2:
            self.policy_id = 2

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.NFD(observation, info)
        elif self.policy_id == 2:
            return self.FFD(observation, info)

    def NFD(self, observation, info):
        list_products = observation["products"]
        list_stocks = observation["stocks"]

        list_empty_acs = self.indices_empty_sort(list_stocks)
        products_sorted = sorted(
            list_products,
            key=lambda p: (-p["size"][0] * p["size"][1], -p["size"][0]),
            reverse=False
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # for stock_idx, stock in enumerate(observation["stocks"]):
        for stock_idx in list_empty_acs:
            stock = list_stocks[stock_idx]
            for product in products_sorted:
                prod_size = product["size"]
                quantity = product["quantity"]

                if quantity > 0:
                    possible_sizes = [prod_size, prod_size[::-1]]
                    for size in possible_sizes:
                        prod_w, prod_h = size

                        stock_w, stock_h = self._get_stock_size_(stock)
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for pos_x in range(stock_w - prod_w + 1):
                                for pos_y in range(stock_h - prod_h + 1):

                                    if self._can_place_(stock, (pos_x, pos_y), prod_size):
                                        return {"stock_idx": stock_idx, "size": size, "position": (pos_x, pos_y)}

        # If no valid placement found, raise an error (this case should be handled externally)
        raise ValueError("No valid placement found for the given products.")

    def get_empty(self, stock):
        return np.sum(stock == -1)

    def indices_empty_sort(self, stocks):
        size_empty = np.array([self. get_empty(stock) for stock in stocks])
        sorted_indices = np.argsort(size_empty)
        return sorted_indices

    def FFD(self, observation, info):
        list_products = observation["products"]
        list_stocks = observation["stocks"]

        list_acs = self.indices_stock_sort(list_stocks)
        products_sorted = sorted(
            list_products,
            key=lambda p: (p["size"][0] * p["size"][1], p["size"][0]),
            reverse=True
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for product in products_sorted:
            prod_size = product["size"]
            quantity = product["quantity"]

            if quantity > 0:
                possible_sizes = [prod_size, prod_size[::-1]]
                for size in possible_sizes:
                    prod_w, prod_h = size

                    # for stock_idx, stock in enumerate(observation["stocks"]):
                    for stock_idx in list_acs:
                        stock = list_stocks[stock_idx]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        if stock_w >= prod_w and stock_h >= prod_h:

                            if stock_w > stock_h:
                                for pos_x in range(stock_w - prod_w + 1):
                                    for pos_y in range(stock_h - prod_h + 1):

                                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                                            return {"stock_idx": stock_idx, "size": size, "position": (pos_x, pos_y)}
                            elif stock_w <= stock_h:
                                for pos_y in range(stock_h - prod_h, -1, -1):
                                    for pos_x in range(stock_w - prod_w + 1):
                                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                                            return {"stock_idx": stock_idx, "size": size, "position": (pos_x, pos_y)}

        # If no valid placement found, raise an error (this case should be handled externally)
        raise ValueError("No valid placement found for the given products.")

    def get_size(self, stock):
        return np.sum(stock != -2)

    def indices_stock_sort(self, stocks):
        size_empty = np.array([-self. get_size(stock) for stock in stocks])
        sorted_indices = np.argsort(size_empty)
        return sorted_indices
