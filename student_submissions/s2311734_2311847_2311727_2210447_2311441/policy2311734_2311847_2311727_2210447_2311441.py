from policy import Policy
import random
import numpy as np 
import math


class Policy2311734_2311847_2311727_2210447_2311441(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here
        self.policy_idx = policy_id
        if policy_id == 1:
            self.stock_idx = 20
        elif policy_id == 2:
            self.initial_temperature = 1000  
            self.cooling_rate = 0.95  
            self.min_temperature = 1e-3  
            self._previous_score = float("inf")  

    def get_action(self, observation, info):
        # Student code here
        if self.policy_idx == 1:
            list_prods = observation["products"]

            check = True
            #dieu kien dung: khi product khong con
            for prod in list_prods:
                if prod["quantity"] > 0:
                    check = False
                    break
                else:
                    check = True

            if check:
                return {"stock_idx": -1, "size": [0,0], "position": (0, 0)}
            
            is_cutted = False
            prod_size = [0,0]
            pos_x, pos_y = 0, 0
            sorted_products = sorted(list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
            for prod in sorted_products:
                if prod["quantity"] > 0:
                    stock = observation["stocks"][self.stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    is_cutted = True
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            break
            
            if is_cutted:
                return{"stock_idx": self.stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
            else:
                self.stock_idx = self.stock_idx+1
                return self.get_action(observation, info)
        elif self.policy_idx == 2:
            current_action = self._random_action(observation)
            current_score = self._evaluate_action(current_action, observation, info)
            
            temperature = self.initial_temperature

            while temperature > self.min_temperature:
                new_action = self._mutate_action(current_action, observation)
                new_score = self._evaluate_action(new_action, observation, info)

                if self._accept_action(current_score, new_score, temperature):
                    current_action = new_action
                    current_score = new_score

                temperature *= self.cooling_rate

            return current_action
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    def _random_action(self, observation):
        list_prods = observation["products"]
        stocks = observation["stocks"]
        valid_stocks = self._filter_valid_stocks(stocks, list_prods)
        if not valid_stocks:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        stock_idx, stock, prod = valid_stocks[random.randint(0, len(valid_stocks) - 1)]
        prod_size = prod["size"]
        possible_positions = self._find_possible_positions(stock, prod_size)
        if possible_positions:
            pos_x, pos_y = possible_positions[random.randint(0, len(possible_positions) - 1)]
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

#----------------------------------------------------------------------------------

    def _filter_valid_stocks(self, stocks, list_prods):
        valid_stocks = []
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)  
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_w, prod_h = prod["size"]
                    if stock_w >= prod_w and stock_h >= prod_h:
                        valid_stocks.append((stock_idx, stock, prod))
                        break  
        return valid_stocks
#----------------------------------------------------------------------------------
    def _find_possible_positions(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        possible_positions = []
        for x in range(stock_w - prod_size[0] + 1):
            for y in range(stock_h - prod_size[1] + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    empty_space = self._calculate_empty_space(stock, x, y, prod_size)
                    possible_positions.append(((x, y), empty_space))

        possible_positions.sort(key=lambda x: x[1])

        return [pos for pos, _ in possible_positions]

#----------------------------------------------------------------------------------
    def _calculate_overlap(self, stock, x, y, prod_size):

        stock_w, stock_h = self._get_stock_size_(stock)
        overlap = 0
        for i in range(x, x + prod_size[0]):
            for j in range(y, y + prod_size[1]):
                if stock[i][j] != -1:  
                    overlap += 1
        return overlap

#----------------------------------------------------------------------------------
    def _calculate_empty_space(self, stock, x, y, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        filled_area = 0
    
        for i in range(x, x + prod_size[0]):
            for j in range(y, y + prod_size[1]):
                if stock[i][j] != -1:  
                    filled_area += 1
        stock_area = stock_w * stock_h
        return stock_area - filled_area
#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

    def _evaluate_action(self, action, observation, info):
        stock_idx = action["stock_idx"]
        pos_x, pos_y = action["position"]
        prod_size = action["size"]


        if stock_idx < 0 or pos_x < 0 or pos_y < 0:
            return float("inf")  # Hành động không hợp lệ sẽ bị phạt nặng

        stock = observation["stocks"][stock_idx]
        

        if not self._can_place_(stock, (pos_x, pos_y), prod_size):
            return float("inf")  # Không thể đặt sản phẩm sẽ bị phạt


        stock_w, stock_h = self._get_stock_size_(stock)
        stock_area = stock_w * stock_h
        prod_area = prod_size[0] * prod_size[1]
        remaining_area = stock_area - prod_area

        total_wasted_area = self._calculate_total_wasted_area(observation)
        wasted_ratio_in_stock = remaining_area / stock_area
        penalty_factor = total_wasted_area / (len(observation["stocks"]) * stock_area)  # Ty le dien tich thua
        score = remaining_area * (1 + penalty_factor)
        if wasted_ratio_in_stock > 0.5:  # neu dien tich thua >
            score += 10  
        delta = abs(score - self._previous_score)  # Chênh lệch điểm số so với hành động trước
        T = 1000  # Tham số nhiệt độ trong quá trình giảm phạt
        score *= math.exp(-delta / T)
        self._previous_score = score

        return score

#----------------------------------------------------------------------------------
    def _calculate_total_wasted_area(self, observation):

        total_wasted_area = 0
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_area = stock_w * stock_h
            filled_area = 0
        for product in observation["products"]:
                if product["quantity"] > 0:
                    prod_size = product["size"]
                    filled_area += prod_size[0] * prod_size[1]

        wasted_area = stock_area - filled_area
        total_wasted_area += wasted_area

        return total_wasted_area
    
    def _get_occupied_area(self, stock):
        """Tính toán diện tích đã chiếm dụng trong kho."""
        occupied_area = 0
        stock_w, stock_h = self._get_stock_size_(stock)  # Kích thước kho (width, height)

        # Duyệt qua tất cả các ô trong kho và đếm những ô không phải là -1 (đã có sản phẩm)
        for x in range(stock_w):
            for y in range(stock_h):
                if stock[x][y] != -1:  
                    occupied_area += 1

        return occupied_area
#----------------------------------------------------------------------------------

    def _mutate_action(self, action, observation):
        new_action = action.copy()
        mutation_type = random.choice(["stock", "position"])

        if mutation_type == "stock":
            new_action["stock_idx"] = random.randint(0, len(observation["stocks"]) - 1)
        elif mutation_type == "position":
            stock_idx = new_action["stock_idx"]
            if stock_idx >= 0:
                stock_w, stock_h = self._get_stock_size_(observation["stocks"][stock_idx])
                prod_size = new_action["size"]
                new_action["position"] = (
                    random.randint(0, stock_w - prod_size[0]),
                    random.randint(0, stock_h - prod_size[1]),
                )

        return new_action

   
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


    def _accept_action(self, current_score, new_score, temperature):
        if new_score < current_score:
            return True
        
        # Nếu nhiệt độ thấp quá, đừng chấp nhận các hành động tồi
        if temperature < 1e-3:
            return False
        
        delta = new_score - current_score
        if delta < 0:
            probability = 1  
        else:
            # Chấp nhận với xác suất giảm dần khi chi phí tăng
            probability = math.exp(-delta / temperature)
        if new_score > 1000:  # chi phí quá lớn thì không chấp nhận
            return False

        return random.random() < probability

    # Student code here
    # You can add more functions if needed
