from policy import Policy
import numpy as np

class BottomLeft(Policy):
    def __init__(self):
        # Student code here
        # A default value of whether a product is placed through rotation or not. If its size is kept originally, reverse=0. Or else reverse=1
        self.reverse=0

    def get_action(self, observation, info):
        # Student code here
        productList=observation["products"]
        # Sort the list of products in a descending order based on their widths as the primary criterion, and their heights as the secondary one
        pList=sorted(productList, key=lambda x: (x["size"][0], x["size"][1]), reverse=True)
        stockList=observation["stocks"]
        # Sort the list of stocks in a descending order based on their areas as the primary criterion, and their widths and heights as the secondary and the third ones respectively
        sList=sorted(stockList, key=lambda x: (self._get_stock_size_(x)[0]*self._get_stock_size_(x)[1], self._get_stock_size_(x)[0], self._get_stock_size_(x)[1]), reverse=True)
        # Reset the value of self.reverse
        self.reverse=0
        # The index of the stock sheet that a product is placed on, the size of that product and the position to be placed
        stock_idx=-1
        size=None
        position=None
        next_w=-1
        next_h=-1
        # Delete all the products whose quantity is zero
        pList=[prod for prod in pList if prod["quantity"]>0]
        next_w, next_h=pList[0]["size"]
        if next_w!=-1 and next_h!=-1:
            # The recent first product to be considered in the sorted list of the products
            next_p=pList[0]
            # Go through the sorted list of stock sheets until next_p can be placed
            for i in range(0,len(sList)):
                stock_w, stock_h=self._get_stock_size_(sList[i])
                # Return the first possible position for next_p to be placed
                validpos=self.valid_pos(sList[i],next_p,stock_w,stock_h)
                if validpos!=None:
                    # Define the index of the stock sheet in the original list (unsorted)
                    indices=[j for j, x in enumerate(stockList) if np.array_equal(x, sList[i])]
                    if len(indices)>1:
                        num=-1
                        for j in range(0,i+1):
                            if np.array_equal(sList[j], sList[i]):
                                num+=1
                        stock_idx=indices[num]
                    else:
                        stock_idx=indices[0]
                    # Define rotation possibility
                    if self.reverse==1:
                        size=next_p["size"][::-1]
                    else:
                        size=next_p["size"]
                    position=validpos
                    # Once finished, break the loop
                    break
        return {"stock_idx": stock_idx, "size": size, "position": position}
    
    def valid_pos(self,stock,next_p,stock_w,stock_h):
        width, height=next_p["size"]
        # Due to the position (0,0) is at the top-left corner, the code will start from the bottom-left position of the stock: (0,stock_h) (BL Heuristic)
        for y in range(stock_h-1, -1, -1):
            for x in range(0, stock_w):
                if x<=stock_w-width and y<=stock_h-height:
                    if self._can_place_(stock, (x,y), next_p["size"]):
                        return (x,y)
                # If the product cannot be placed normally, try to rotate it
                if x<=stock_w-height and y<=stock_h-width:
                    if self._can_place_(stock, (x,y), next_p["size"][::-1]):
                        self.reverse=1
                        return (x,y)
        return None

class FFDH(Policy):
    def __init__(self):
        # Student code here
        self.lvl = [[1000 for _ in range(100)] for _ in range(100)] # Create a 2D matrix holds lvls for each stock (hold 100 lvls)

    def get_action(self, observation, info):
        # Student code here
        product_lst = observation["products"]
        sorted_products = sorted(product_lst, key = lambda x: (x["size"][1], x["size"][0]), reverse = True) # Sort height of products, put the product that is tallest to to stock first
        # The index of the stock sheet that a product is placed on, the size of that product and the position to be placed
        stock_index = -1
        pos_x = None
        pos_y = None
        for product in sorted_products:
            if product["quantity"] > 0:
                product_size = product["size"]
                product_width, product_height = product_size
                for i, stock in enumerate(observation["stocks"]):
                    stock_width, stock_height = self._get_stock_size_(stock) # Get stock width and height
                    if stock_width < product_width or stock_height < product_height: # Check if the product does fin one or both demension or not, if not go to next stock
                        continue                    
                    # example: if product has the height of 2 and stock has the height of 5, the placement y coordinate should be 3 (stock height starts at 0 and plaecment coordinate is top left)
                    # 0
                    # 1
                    # 2
                    # 3 <- precisely here so it can take up 2 rows: 3 and 4 (product heigth is 2)
                    # 4
                    y = stock_height - product_height
                    next_lvl = 1
                    while y >= 0: # Given that placement coordinate of a product is top left so we go from the bottom of the stock (stock_height) to the top (0)
                        for x in range(stock_width - product_width + 1):
                            if self._can_place_(stock, (x, y), product_size):
                                pos_x = x
                                pos_y = y
                                stock_index = i
                                if pos_x == 0: # Create new level if product cannot fit in the current level (after run on the x axis)
                                    self.lvl[i][next_lvl] = pos_y
                                break                            
                        if pos_x is not None and pos_y is not None:
                            break
                        y = self.lvl[i][next_lvl] - product_height # Jump to next lvl
                        next_lvl = next_lvl + 1
                    if pos_x is not None and pos_y is not None:
                        break
            if pos_x is not None and pos_y is not None:
                break
        return {"stock_idx": stock_index, "size": product_size, "position": (pos_x, pos_y)}