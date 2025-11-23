import numpy as np
from policy import Policy

class Policy2053155_2053048(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = 1
        elif policy_id == 2:
            self.policy = 2

        self.policy_id = policy_id
        self.last_prod = None
        self.last_stock_idx = -1

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.first_fit(observation, info)
        elif self.policy_id == 2:
            return self.best_fit(observation, info)

    def first_fit(self, observation, info):
        products = sorted(observation["products"], key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
        stocks = list(enumerate(observation["stocks"]))
        stocks.sort(key=lambda stock: np.sum(stock[1] != -2), reverse=True)
        return self.cutting(products, stocks, info)

    def best_fit(self, observation, info):
        products = sorted(observation["products"], key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
        stocks = list(enumerate(observation["stocks"]))
        
        # Filter and sort stocks more effectively
        filter_list = [stock for stock in stocks if np.sum(stock[1] >= 0)]
        filter_list.sort(key=lambda stock: np.sum(stock[1] >= 0), reverse=True)

        remain_list = [stock for stock in stocks if np.all(stock[1] < 0)]
        remain_list.sort(key=lambda stock: np.sum(stock[1] != -2), reverse=True)

        stocks = filter_list + remain_list
        return self.cutting(products, stocks, info)

    def cutting(self, products, stocks, info):
        stock_idx = -1
        prod_size = [0, 0]
        last_prod_size = [0, 0]
        pos_x, pos_y = None, None
        cutted = False
        jump_check = True

        if self.last_prod is not None:
            last_prod_size = self.last_prod["size"]

        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in stocks:
                    stock_w, stock_h = self._get_stock_size(stock)

                    # Avoid cutting the same stock-product combination again
                    if self.last_stock_idx == i and jump_check:
                        jump_check = False

                    if np.array_equal(last_prod_size, prod_size) and jump_check:
                        continue

                    current_area_left = np.sum(stock == -1)
                    if current_area_left < prod_size[0] * prod_size[1]:
                        continue

                    # Check if the product can fit in the stock
                    if stock_w >= prod_size[0] and stock_h >= prod_size[1] and not cutted:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    cutted = True
                                    break
                            if cutted:
                                break

                    if stock_w >= prod_size[1] and stock_h >= prod_size[0] and not cutted:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_size[1] + 1):
                            for y in range(stock_h - prod_size[0] + 1):
                                if self._can_place(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    cutted = True
                                    break
                            if cutted:
                                break

                    if cutted:
                        stock_idx = i
                        self.last_prod = prod
                        self.last_stock_idx = stock_idx
                        break

                if cutted:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


    def _get_stock_size(self, stock):
        return stock.shape

    def _can_place(self, stock, position, prod_size):
        x, y = position
        prod_w, prod_h = prod_size
        if x + prod_w > stock.shape[0] or y + prod_h > stock.shape[1]:
            return False
        return np.all(stock[x:x+prod_w, y:y+prod_h] == -1)
