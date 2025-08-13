class Node:
    """
    Represents a node in the discrete search space used for path planning.

    Attributes:
        x (float): X-coordinate of the node in world or grid space.
        y (float): Y-coordinate of the node in world or grid space.
        cost (float): Edge cost to reach this node from its parent.
        cost_to_come (float): Accumulated cost from the start node to this node.
        cost_to_go (float): Heuristic estimate of cost from this node to the goal.
        total_cost (float): Sum of cost_to_come and cost_to_go.
        children (list[Node]): List of child nodes connected to this node.
        parent (Node | None): Reference to the parent node in the search tree.
    """
    def __init__(self, x, y, cost, children=None):
        """
        Initialize a Node object.

        Args:
            x (float): X-coordinate of the node.
            y (float): Y-coordinate of the node.
            cost (float): Edge cost from parent to this node.
            children (list[Node] | None): Optional list of initial child nodes.
                Defaults to an empty list if None.
        """
        self.cost = cost
        self.x = x
        self.y = y
        self.cost_to_come = 0
        self.cost_to_go = 0 
        self.total_cost = 0
        self.children = [] if children is None else children
        self.parent = None
        
    def add_child(self, child_node):
        """
        Add a child node to this node's children list.

        Args:
            child_node (Node): The child node to connect to this node.
        """
        self.children.append(child_node)
        
    