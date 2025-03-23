from typing import List
import numpy as np
from scipy.spatial import distance

class VRPData:
    """
    Vehicle Routing Problem (VRP) data structure class.
    Maintains the state of a VRP instance including routes, distances, demands, and various similarity metrics.
    """
    def __init__(self, points: np.array, vehicle_capacity: float, node_demand: np.array, routes: List[List[int]]=None, cost: int=None):
        """
        Initialize VRP instance with problem data and compute necessary matrices.
        
        Args:
            points: np.array - Coordinates of nodes (first node is depot)
            vehicle_capacity: float - Maximum capacity of each vehicle
            node_demand: np.array - Demand of each node (first node is depot with 0 demand)
            routes: List[List[int]] - Initial routes, each starting and ending at depot (optional)
            cost: int - Initial solution cost (optional, will be computed if not provided)
        """
        self.routes = routes if routes is not None else []
        self.customers = [node for route in self.routes for node in route if node != 0]
        self.unassigned_customers = []  # Start with empty unassigned customers - will be populated by destroy actions
        self.vehicle_capacity = vehicle_capacity
        self.points = points
        self.num_nodes = points.shape[0]
        self.node_demand = node_demand
        self.distance_matrix = self.compute_distance_matrix()
        self.nearest_neighbors = np.argsort(self.distance_matrix, axis=1)
        self.nearest_demands = self.compute_demand_similarity()
        
        if not cost:
            self.cost = self.compute_cost()

    def compute_cost(self) -> float:
        """
        Compute total cost of all routes by summing distances between consecutive nodes.
        
        Returns:
            float - Total distance cost of all routes
        """
        cost = 0
        for route in self.routes:
            # Sum distances between consecutive nodes in route
            cost += sum(self.distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        return cost

    def add_routes(self, routes: List[List[int]]) -> None:
        """
        Update the current solution with new routes.
        
        Args:
            routes: List[List[int]] - New routes to replace current solution
        """
        self.routes = routes

    def compute_distance_matrix(self) -> np.array:
        """
        Compute Euclidean distance matrix between all pairs of nodes.
        Sets diagonal elements to infinity to prevent self-loops.
        
        Returns:
            np.array - nxn matrix where entry (i,j) is Euclidean distance between nodes i and j
                      diagonal elements are set to infinity
        """
        dist_matrix = distance.cdist(self.points, self.points, 'euclidean')
        np.fill_diagonal(dist_matrix, float('inf'))
        return dist_matrix

    def compute_demand_similarity(self) -> np.array:
        """
        Compute demand similarity matrix between all pairs of nodes.
        Similarity is measured as absolute difference between node demands.
        Sets diagonal elements and depot connections to infinity.
        Only computes similarities between non-depot nodes.
        
        Returns:
            np.array - Sorted indices of nodes based on demand similarity
                      (each row i contains indices of other nodes sorted by demand similarity to node i)
                      For depot (row 0), all similarities are infinity
        """
        demand_similarity = np.full((self.num_nodes, self.num_nodes), float('inf'))
        
        # Calculate all demand differences for non-depot nodes
        for i in range(1, self.num_nodes):  # Start from 1 to skip depot
            for j in range(1, self.num_nodes):  # Start from 1 to skip depot
                if i != j:
                    demand_similarity[i,j] = abs(self.node_demand[i] - self.node_demand[j])
        
        return np.argsort(demand_similarity, axis=1)

    def compute_route_similarity(self) -> np.array:
        """
        Compute node-to-node similarity based on route sequence proximity.
        For each pair of nodes in the same route, calculates their sequence distance.
        Nodes in different routes or same node get infinity distance.
        
        Returns:
            np.array - Sorted indices of nodes based on route sequence similarity
                      (each row i contains indices of other nodes sorted by sequence distance to node i)
        Example:
            For route [0,1,2,3,0]:
            - Nodes 1 and 2 have distance 1 (adjacent)
            - Nodes 1 and 3 have distance 2 (two positions apart)
            - Nodes in different routes have distance infinity
        """
        route_similarity = np.full((self.num_nodes, self.num_nodes), float('inf'))
        
        for route in self.routes:
            # Skip depot connections
            route = [n for n in route if n != 0]
            
            # For each pair of positions in route
            for i, node1 in enumerate(route):
                for j, node2 in enumerate(route):
                    if node1 != node2:
                        # Similarity is sequence distance (smaller means more similar)
                        sequence_distance = abs(i - j)
                        route_similarity[node1, node2] = sequence_distance
                        route_similarity[node2, node1] = sequence_distance
        
        return np.argsort(route_similarity, axis=1)

    def compute_worst_nodes(self) -> np.array:
        """
        Compute potential cost improvement for removing each node.
        For each node (except depot), calculates the cost difference between:
        - Current cost: distance from previous to node plus node to next
        - New cost: direct distance from previous to next
        
        Returns:
            np.array - Sorted indices of nodes based on potential cost improvement
                      (nodes whose removal would save more cost come first)
                      Negative improvement means removing the node would increase cost
        """
        improvement = np.full((self.num_nodes), -float('inf'))
        for route in self.routes:
            for i, node in enumerate(route):
                if 0 < i < len(route)-1:  # Skip depot and endpoints
                    prev = route[i-1]
                    next = route[i+1]
                    current_cost = self.distance_matrix[prev,node] + self.distance_matrix[node,next]
                    new_cost = self.distance_matrix[prev,next]
                    improvement[node] = current_cost - new_cost  # Positive means removal improves solution
        return np.argsort(-improvement)  # Sort descending so best improvements come first

    def update_cost(self) -> None:
        """
        Update the current solution with a new route and recompute cost.
        
        Args:
            new_route: List[List[int]] - New route to replace current solution
        """
        self.cost = self.compute_cost()
        return self.cost
        




        
    