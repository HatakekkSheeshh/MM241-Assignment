from policy import Policy

class Policy2311237_2252726_2012646_2420001_2311191(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        print(f"Initialized Policy with ID: {self.policy_id}")
        if policy_id == 1:
            self.p=0
            self.numberOfProducts=0
            self.stocklist=[]
        elif policy_id == 2:
            self.last_used_stock_idx = 0

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.policy1(observation, info)
        elif self.policy_id == 2:
            return self.policy2(observation, info)

    # Student code here
    def policy1(self, observation, info):
        list_prods = observation["products"]
        stock_idx=-1
        prod_size=[0, 0]
        pos_x, pos_y = 0, 0
        stocks = observation["stocks"]
        self.p=self.p+1
        if (self.numberOfProducts==0):
            for prod in list_prods:
                self.numberOfProducts=prod['quantity']+self.numberOfProducts
                
        if (self.stocklist==[]):
            i=0
            for stock in stocks: 
                a={"stock_area":None,"index":None}
                stock_w,stock_h=self._get_stock_size_(stock)
                a["stock_area"]=stock_w*stock_h
                a["index"]=i
                i=i+1
                self.stocklist.append(a)
        observation["products"]=sorted(observation["products"],key=lambda x:x['size'][0]+x['size'][1],reverse=True)
        self.stocklist=sorted(self.stocklist,key=lambda x:x['stock_area'], reverse=False)
        
        index=0
        
        for prod in observation["products"]:
            if prod["quantity"]>0:
                prod_size=prod["size"]

                for i, a in enumerate(self.stocklist):
                    stock_w, stock_h = self._get_stock_size_(observation["stocks"][a['index']])
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        prod_size=[prod_h,prod_w]
                        prod_w, prod_h = prod_size
                        if stock_w < prod_w or stock_h < prod_h:
                           prod_size=[prod_h,prod_w]
                           continue

                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(observation["stocks"][a['index']], (x, y), prod_size):
                                index=i
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = a['index']
                        break
                    else: 
                        if prod_w<=stock_h and prod_h<=stock_w:
                           prod_size=[prod_h,prod_w]
                           for x in range(stock_w - prod_w + 1):
                               for y in range(stock_h - prod_h + 1):
                                   if self._can_place_(observation["stocks"][a['index']], (x, y), prod_size):
                                      index=i
                                      pos_x, pos_y = x, y
                                   break
                               if pos_x is not None and pos_y is not None:
                                  break
                           if pos_x is not None and pos_y is not None:
                               stock_idx=a['index']
                               break
                           else:
                               prod_size=[prod_h,prod_w]

                if pos_x is not None and pos_y is not None:
                    break
        c={"stock_area":None,"index":None}
        c["stock_area"]=self.stocklist[index]['stock_area']-prod_w*prod_h
        c["index"]=self.stocklist[index]['index']
        del self.stocklist[index]
        for z, b in enumerate(self.stocklist):
            if (b['stock_area']>=c['stock_area'])and(z+1>=len(self.stocklist) or(self.stocklist[z+1]['stock_area']<c['stock_area'])):
                self.stocklist.insert(i,c)
                break

        if (self.p==self.numberOfProducts):
            self.stocklist.clear()
            self.stocklist=[]
            self.p=0
            self.numberOfProducts=0
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


    def policy2(self, observation, info):
        list_prods = sorted(
            observation["products"],
            key=lambda p: p["size"][1],  # Sort by height (size[1])
            reverse=True
        )
        stocks = observation["stocks"]

        num_stocks = len(stocks)

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                stock_idx = self.last_used_stock_idx
                checked_stocks = 0

                while checked_stocks < num_stocks:
                    stock = stocks[stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    # Found a valid position in the current stock
                                    self.last_used_stock_idx = stock_idx  # Update last used stock index
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }
                    stock_idx = (stock_idx + 1) % num_stocks
                    checked_stocks += 1

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
    # You can add more functions if needed
