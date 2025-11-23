import numpy as np
from policy import Policy
from abc import abstractmethod

# define infinte value
INF = 1e9


class BestFit(Policy):
    def __init__(self):
        pass

    def get_action(self, obeservation, info):
        # intinialize the action
        list_products = obeservation["products"]  # input
        BestStockIdx, BestPos, BestSize = -1, None, None  # output

        # Sort the products by area
        # The biggest product is the first
        # The core idea is to pick the biggest product first and then the smallest can be placed in the small between the big products
        sorted_list = sorted(
            (prod for prod in list_products if prod["quantity"] > 0), key=lambda x: x["size"][0]*x["size"][1], reverse=True)

        # Interate over the sorted products
        for prod in sorted_list:
            # Pick a product that has quality > 0
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                BestUnsedArea = INF  # The best unused area is the smallest

                # Iterate over the stocks
                for idx, stock in enumerate(obeservation["stocks"]):
                    stockW, stockH = self._get_stock_size_(stock)
                    prodW, prodH = prod_size

                    # Main action
                    for rotated in [False, True]:  # Rotate the product
                        if rotated:
                            prod_size = prod_size[::-1]
                        if stockW >= prod_size[0] and stockH >= prod_size[1]:
                            position = self.getPosition(stock, prod_size)
                            if position:
                                unused_area = (stockW * stockH) - \
                                    (prod_size[0] * prod_size[1])
                                if unused_area < BestUnsedArea:
                                    BestUnsedArea = unused_area
                                    BestStockIdx, BestPos, BestSize = idx, position, prod_size

        # Return the action
        return {"stock_idx": BestStockIdx, "size": BestSize, "position": BestPos}

    def getPosition(self, stock, ProductSize):
        '''
        Find the pos from left to right, from high to low
        '''
        prodW, prodH = ProductSize
        stockW, stockH = self._get_stock_size_(stock)

        for x in range(stockW - prodW + 1):
            for y in range(stockH - prodH + 1):
                if self._can_place_(stock, (x, y), ProductSize):
                    return x, y


class FirstFit(Policy):
    def __init__(self):

        self.m_used_surface = 0
        self.m_filled_surface = 0
        self.m_used_stock = 0

        self.m_sorted_stock_index = 0
        self.m_sorted_product_index = 0

        pass

    def reset(self):

        self.m_used_surface = 0
        self.m_filled_surface = 0
        self.m_used_stock = 0

        self.m_sorted_stock_index = 0
        self.m_sorted_product_index = 0

        pass

    def paint(self, stocks, products, stock_idx, prod_idx, position, prod_size):
        width, height = prod_size
        x, y = position

        stock = stocks[stock_idx]
        stock[x: x + width, y: y + height] = prod_idx
        products[prod_idx]["quantity"] -= 1

    def evaluation(self):
        print("used surface:   ", self.m_used_surface)
        print("filled surface: ", self.m_filled_surface)
        print("filled percent: ", self.m_filled_surface *
              100 / self.m_used_surface, "%")
        print("waste perent:   ", 100 - self.m_filled_surface *
              100 / self.m_used_surface, "%")
        print("used stocks:    ", self.m_used_stock)
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]

        # Descending
        sorted_products = sorted(
            list_prods, key=lambda product: product['size'][0] * product['size'][1], reverse=True)
        # sorted_products = sorted(list_prods, key=lambda product: product['size'][0], reverse=True)
        product_indies = []
        if (self.m_sorted_product_index == 0):
            print(sorted_products)
            for s_st in range(len(sorted_products)):
                for st in range(len(list_prods)):
                    if (np.shape(list_prods[st]['size']) == np.shape(sorted_products[s_st]['size'])) and (np.all(list_prods[st]['size'] == sorted_products[s_st]['size'])):
                        product_indies.append(st)
            self.m_sorted_product_index = product_indies
        else:
            product_indies = self.m_sorted_product_index

        sorted_stocks = sorted(list_stocks, key=lambda stock: np.sum(np.any(
            stock != -2, axis=1)) * np.sum(np.any(stock != -2, axis=0)), reverse=True)
        stock_indies = []
        if (self.m_sorted_stock_index == 0):

            for s_st in range(len(sorted_stocks)):
                for st in range(len(list_stocks)):
                    if (np.shape(list_stocks[st]) == np.shape(sorted_stocks[s_st])) and (np.all(list_stocks[st] == sorted_stocks[s_st])):
                        stock_indies.append(st)
            self.m_sorted_stock_index = stock_indies
        else:
            stock_indies = self.m_sorted_stock_index

        prod_size = [0, 0]
        prod_idx = -1
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for pr_idx in product_indies:
            prod = list_prods[pr_idx]
            if prod["quantity"] > 0:
                # Loop through all stocks
                for st_idx in stock_indies:
                    stock = list_stocks[st_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod['size']

                    # evaluate
                    used = np.any(stock >= 0)
                    surface = stock_w * stock_h
                    filled = np.sum(stock >= 0)

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                prod_size = (prod_w, prod_h)
                                print(prod_size)

                                if (not used):
                                    self.m_used_surface += + surface
                                    self.m_used_stock += 1

                                prod_surface = prod_w * prod_h
                                self.m_filled_surface += prod_surface
                                filled += prod_surface

                                pos_x, pos_y = x, y
                                prod_idx = pr_idx
                                self.evaluation()
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = st_idx
                        break
                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    # Student code here
    # You can add more functions if needed


class Policy2312596_2313097_2311318_2312954_2312791(Policy):
    def __init__(self, policy_id=1):
        # Kiểm tra policy_id có phải 1 hoặc 2 không
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = BestFit()  # Chọn Policy1 nếu policy_id = 1
        elif policy_id == 2:
            self.policy = FirstFit()  # Chọn Policy2 nếu policy_id = 2

    def get_action(self, observation, info):
        # Gọi hàm get_action từ policy đã chọn
        return self.policy.get_action(observation, info)
