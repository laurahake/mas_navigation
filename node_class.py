class Node:
    def __init__(self, x, y, cost, children=None):
        self.cost = cost
        self.x = x
        self.y = y
        self.cost_to_come = 0
        self.cost_to_go = 0 
        self.total_cost = 0
        self.children = [] if children is None else children
        self.parent = None
        
    def add_child(self, child_node):
        self.children.append(child_node)
        
    