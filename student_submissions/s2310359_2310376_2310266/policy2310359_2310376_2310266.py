from policy import Policy


class Policy2310359_2310376_2310266(Policy):
    def __init__(self):
        # Student code here
        self.skylines = []
        self.stock_used = 0

        self.stocks = []
        self.area_used = 0
        self.area_used_stock = 0
        pass

    def get_action(self, observation, info):
        # Student code here
        if info['filled_ratio'] == 0.0:
            # the env have been reseted
            self.init_skyline_set(observation["stocks"])

        list_prods = observation["products"]
        list_prods = sorted(list_prods, key=lambda x: -(x["size"][0] * x["size"][1]))

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] == 0: continue

            prod_size = prod["size"]
            if (prod_size[0] < prod_size[1]): prod_size[0], prod_size[1] = prod_size[1], prod_size[0]

            waste_area, (pos_x, pos_y) = 1e18, (-1, -1)
            stock_idx_choice = -1
            should_rotate = False

            for i, (stock, stock_idx) in enumerate(self.stocks):
                if i >= self.stock_used: break

                current_waste_area, (tmp_x, tmp_y), tmp_rotate = self.calculate_minimum_local_waste(prod, stock, stock_idx)
                if (current_waste_area < waste_area):
                    waste_area = current_waste_area
                    pos_x, pos_y = tmp_x, tmp_y
                    stock_idx_choice = stock_idx
                    should_rotate = tmp_rotate
                    break

            # Use new stock
            while waste_area == 1e18:
                self.stock_used += 1
                stock_w, stock_h = self._get_stock_size_(self.stocks[self.stock_used - 1][0])
                self.area_used_stock += stock_w * stock_h

                current_waste_area, (tmp_x, tmp_y), tmp_rotate = self.calculate_minimum_local_waste(prod, self.stocks[self.stock_used - 1][0], self.stocks[self.stock_used - 1][1])
                if (current_waste_area < 1e18):
                    waste_area = current_waste_area
                    pos_x, pos_y = tmp_x, tmp_y
                    stock_idx_choice = self.stocks[self.stock_used - 1][1]
                    should_rotate = tmp_rotate
                    break
            
            if (should_rotate): prod["size"][0], prod["size"][1] = prod["size"][1], prod["size"][0]
            self.place_product(prod, (pos_x, pos_y), observation["stocks"][stock_idx_choice], stock_idx_choice)
            place = {"stock_idx": stock_idx_choice, "size": prod_size, "position": (pos_x, pos_y)}
            print(f'place: {place}')

            self.area_used += prod["size"][0] * prod["size"][1]

            print(1.0 - self.area_used / self.area_used_stock)
            return place 

    # Student code here
    # You can add more functions if needed

    # CURRENT OPTIMAL STRATEGY:
    # Minimize local waste
    # Stock sorted in size
    # Rotate when it makes less waste area
    # Product sorted by area
    # Used more stock when needed

    # Consider height ???
    # Maximize fitness level ???

    # Skyline is a set of 3-int tuple Skyline(x_start, x_end, height)
    class SkylinePart():
        def __init__(self, x_start, x_end, height) -> None:
            self.x_start = x_start
            self.x_end = x_end
            self.height = height

        def __str__(self):
            return f'SkylinePart({self.x_start}, {self.x_end}, {self.height})'

        def _intersect_(self, prod_size, prod_pos) -> bool:
            # intersect here means positive common areas, which makes this placement invalid

            prod_x_start, prod_y = prod_pos
            prod_w, prod_h = prod_size
            prod_x_end = prod_x_start + prod_w

            if (prod_x_end <= self.x_start or prod_x_start >= self.x_end): return False

            if (prod_y >= self.height): return False
            
            return True
        
        def _calculate_local_waste_(self, prod_size, prod_pos) -> int:
            prod_x_start, prod_y = prod_pos
            prod_w, prod_h = prod_size
            prod_x_end = prod_x_start + prod_w

            if (prod_x_end <= self.x_start or prod_x_start >= self.x_end): return 0

            return (prod_y - self.height) * (min(prod_x_end, self.x_end) - max(prod_x_start, self.x_start))

    #####################

    
    def init_skyline_set(self, stocks):
        # this will initialize the skyline set
        self.skylines = []

        self.stocks = [(stock, i) for i, stock in enumerate(stocks)]
        print(self.stocks[:10])
        self.stocks = sorted(self.stocks, key= lambda x : -(self._get_stock_size_(x[0])[0] * self._get_stock_size_(x[0])[1]))

        for stock, i in self.stocks:
            self.skylines += [[Policy2310359.SkylinePart(0, self._get_stock_size_(stock)[0], 0)]]

    def calculate_minimum_local_waste(self, product, stock, stock_idx):
        # Current idea: 
        # Each time we add the product to make smallest local waste
        # This function calculates local waste
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product["size"]

        WONT_USE = 1e18

        waste_area = WONT_USE
        placement_choice = (-1, -1)
        should_rotate = False

        # First, we try placing on the leftmost of each part of skyline
        for part_idx, part in enumerate(self.skylines[stock_idx]):
            # now have to check validness
            pos_x, pos_y = part.x_start, part.height

            if pos_x < 0 or pos_x + prod_w > stock_w: continue
            if pos_y < 0 or pos_y + prod_h > stock_h: continue
            if not self._can_place_(stock, (pos_x, pos_y), product["size"]): continue
            ok = True
            for part in self.skylines[stock_idx]:
                if part._intersect_(product["size"], (pos_x, pos_y)):
                    ok = False
                    break
            if not ok: continue

            current_waste_area = 0
            for each_part in self.skylines[stock_idx]:
                current_waste_area += each_part._calculate_local_waste_((prod_w, prod_h), (pos_x, pos_y))

            if waste_area > current_waste_area:
                waste_area = current_waste_area
                placement_choice = (pos_x, pos_y)
                should_rotate = False

        # rightmost on each part (kinda similar)
        for part_idx, part in enumerate(self.skylines[stock_idx]):
            # now have to check validness
            pos_x, pos_y = part.x_end - prod_w, int(part.height)

            if pos_x < 0 or pos_x + prod_w > stock_w: continue
            if pos_y < 0 or pos_y + prod_h > stock_h: continue
            if not self._can_place_(stock, (pos_x, pos_y), product["size"]): continue
            ok = True
            for part in self.skylines[stock_idx]:
                if part._intersect_(product["size"], (pos_x, pos_y)):
                    ok = False
                    break
            
            if not ok: continue

            current_waste_area = 0
            for each_part in self.skylines[stock_idx]:
                current_waste_area += each_part._calculate_local_waste_((prod_w, prod_h), (pos_x, pos_y))

            if waste_area > current_waste_area:
                waste_area = current_waste_area
                placement_choice = (pos_x, pos_y)
                should_rotate = False

        ################ ROTATE OBJECT ########################
        # First, we try placing on the leftmost of each part of skyline
        prod_w, prod_h = prod_h, prod_w

        for part_idx, part in enumerate(self.skylines[stock_idx]):
            # now have to check validness
            pos_x, pos_y = part.x_start, part.height

            if pos_x < 0 or pos_x + prod_w > stock_w: continue
            if pos_y < 0 or pos_y + prod_h > stock_h: continue
            if not self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)): continue
            ok = True
            for part in self.skylines[stock_idx]:
                if part._intersect_((prod_w, prod_h), (pos_x, pos_y)):
                    ok = False
                    break
            if not ok: continue

            current_waste_area = 0
            for each_part in self.skylines[stock_idx]:
                current_waste_area += each_part._calculate_local_waste_((prod_w, prod_h), (pos_x, pos_y))

            if waste_area > current_waste_area:
                waste_area = current_waste_area
                placement_choice = (pos_x, pos_y)
                should_rotate = True

        # rightmost on each part (kinda similar)
        for part_idx, part in enumerate(self.skylines[stock_idx]):
            # now have to check validness
            pos_x, pos_y = part.x_end - prod_w, int(part.height)

            if pos_x < 0 or pos_x + prod_w > stock_w: continue
            if pos_y < 0 or pos_y + prod_h > stock_h: continue
            if not self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)): continue
            ok = True
            for part in self.skylines[stock_idx]:
                if part._intersect_((prod_w, prod_h), (pos_x, pos_y)):
                    ok = False
                    break
            
            if not ok: continue

            current_waste_area = 0
            for each_part in self.skylines[stock_idx]:
                current_waste_area += each_part._calculate_local_waste_((prod_w, prod_h), (pos_x, pos_y))

            if waste_area > current_waste_area:
                waste_area = current_waste_area
                placement_choice = (pos_x, pos_y)
                should_rotate = True
        
        return int(waste_area), placement_choice, should_rotate

    def place_product(self, product, pos, stock, stock_idx):
        new_skyline = []

        prod_w, prod_h = product["size"]

        pos_x, pos_y = pos
        x_start = int(pos_x)
        x_end = int(x_start + prod_w)

        stock_w, stock_h = self._get_stock_size_(stock)
        height = pos_y + prod_h

        new_skyline += [Policy2310359.SkylinePart(x_start, x_end, height)]

        for other_part in self.skylines[stock_idx]:
            if other_part.x_end <= x_start or other_part.x_start >= x_end:
                # not interset
                new_skyline += [other_part]
                continue
                
            if x_start <= other_part.x_start and other_part.x_end <= x_end:
                continue

            if other_part.x_start < x_start:
                new_skyline += [Policy2310359.SkylinePart(other_part.x_start, x_start, other_part.height)]
            else:
                new_skyline += [Policy2310359.SkylinePart(x_end, other_part.x_end, other_part.height)]


        new_skyline = sorted(new_skyline, key=lambda x: x.x_start)

        self.skylines[stock_idx] = new_skyline

        print(f'stock_idx: {stock_idx}')
        for i in self.skylines[stock_idx]:
            print(i,end=',')
        print('')