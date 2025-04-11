from copy import deepcopy
from typing import List, Tuple
from torch_geometric.data import Data
from itertools import chain
import random
import numpy as np
from vrp_data import VRPData

class Destroy:
    """
    Base class for implementing destroy operators in the ALNS algorithm.
    Destroy operators remove customers from routes to create partial solutions
    that can be repaired by repair operators. The removed customers are tracked
    in the VRPData state for later repair operations.
    """
    def remove_customers(self, state: VRPData, customers_to_remove: List[int]) -> Tuple[VRPData, List[List[int]], List[List[int]],float]:
        """
        Removes specified customers from the given state's routes.
        
        Args:
            state: VRPData - The current state of the VRP instance
            customers_to_remove: List[int] - List of customer indices to remove from routes
            
        Returns:
            Tuple containing:
                - VRPData: Modified state with updated routes
                - List[List[int]]: New edges added (source and target nodes)
                - List[List[int]]: Deleted edges (source and target nodes)
        """
        new_routes = []
        cost_change=0
        for route in state.routes:
            new_route = []
            for node_index,node in enumerate(route):
                if node in customers_to_remove:
                    if len(route) > 3: # if it is just depot-cust-node then there is no edge to add
                        cost_change+=state.distance_matrix[route[node_index+1],route[node_index-1]]
                    cost_change-=state.distance_matrix[node,route[node_index+1]]+state.distance_matrix[node,route[node_index-1]]
                else:
                    new_route.append(node)
            if len(new_route) > 2:  # Keep route if it has more than just depot-depot
                new_routes.append(new_route)
        state.routes = new_routes
        state.unassigned_customers = customers_to_remove
        return state, cost_change

class RandomRemoval(Destroy):
    """
    Random removal destroy operator that randomly selects and removes customers from routes.
    Inherits from the base Destroy class. Updates the VRPData state with removed customers
    for tracking unassigned customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> Tuple[VRPData, List[List[int]], List[List[int]], float]:
        """
        Randomly removes a specified number of customers from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to randomly remove
            
        Returns:
            Tuple containing:
                - VRPData: Modified state with updated routes and unassigned customers
                - List[List[int]]: New edges added (source and target nodes)
                - List[List[int]]: Deleted edges (source and target nodes)
            
        Raises:
            ValueError - If routes are empty (nothing to destroy)
        """
        if not state.routes:
            raise ValueError("Cannot destroy empty routes - no customers to remove")
            
        # Randomly select customers from existing routes
        if len(state.customers) <= num_customers_to_remove:
            customers_to_remove = state.customers
        else:
            customers_to_remove = random.sample(state.customers, k=num_customers_to_remove)
        
        # Remove selected customers from routes and update state
        return self.remove_customers(state, customers_to_remove)

class DemandRelatedRemoval(Destroy):
    """
    Destroy operator that removes customers with similar demands.
    Selects a random customer and removes its nearest neighbors based on demand similarity.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> Tuple[VRPData, List[List[int]], List[List[int]],float]:
        """
        Removes customers with similar demands from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on demand similarity
            
        Returns:
            Tuple containing:
                - VRPData: Modified state with updated routes and unassigned customers
                - List[List[int]]: New edges added (source and target nodes)
                - List[List[int]]: Deleted edges (source and target nodes)
        """
        first_customer = random.choice(state.customers)
        customers_to_remove = [first_customer]
        
        # Get nearest neighbors by demand, excluding those already selected
        neighbors = state.nearest_demands[first_customer].tolist()
        for neighbor in neighbors:
            if len(customers_to_remove) >= num_customers_to_remove:
                break
            if neighbor != 0 and neighbor not in customers_to_remove:  # Skip depot and already selected
                customers_to_remove.append(neighbor)
        
        return self.remove_customers(state, customers_to_remove)

class GeographicRelatedRemoval(Destroy):
    """
    Destroy operator that removes geographically close customers.
    Selects a random customer and removes its nearest neighbors based on Euclidean distance.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> Tuple[VRPData, List[List[int]], List[List[int]],float]:
        """
        Removes geographically close customers from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on geographic proximity
            
        Returns:
            Tuple containing:
                - VRPData: Modified state with updated routes and unassigned customers
                - List[List[int]]: New edges added (source and target nodes)
                - List[List[int]]: Deleted edges (source and target nodes)
        """
        first_customer = random.choice(state.customers)
        customers_to_remove = [first_customer]
        
        # Get nearest neighbors by distance, excluding those already selected
        neighbors = state.nearest_neighbors[first_customer].tolist()
        for neighbor in neighbors:
            if len(customers_to_remove) >= num_customers_to_remove:
                break
            if neighbor != 0 and neighbor not in customers_to_remove:  # Skip depot and already selected
                customers_to_remove.append(neighbor)
        return self.remove_customers(state, customers_to_remove)

class RouteRelatedRemoval(Destroy):
    """
    Destroy operator that removes customers that are close to each other in route sequence.
    Selects a random customer and removes its nearest neighbors based on route position.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> Tuple[VRPData, List[List[int]], List[List[int]],float]:
        """
        Removes customers that are close in route sequence from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on route proximity
            
        Returns:
            Tuple containing:
                - VRPData: Modified state with updated routes and unassigned customers
                - List[List[int]]: New edges added (source and target nodes)
                - List[List[int]]: Deleted edges (source and target nodes)
        """
        first_customer = random.choice(state.customers)
        customers_to_remove = [first_customer]
        
        # Get nearest neighbors by route sequence, excluding those already selected
        nearest_neighbors_by_route = state.compute_route_similarity()
        neighbors = nearest_neighbors_by_route[first_customer].tolist()
        for neighbor in neighbors:
            if len(customers_to_remove) >= num_customers_to_remove:
                break
            if neighbor != 0 and neighbor not in customers_to_remove:  # Skip depot and already selected
                customers_to_remove.append(neighbor)
        
        return self.remove_customers(state, customers_to_remove)
        

class GreedyRemoval(Destroy):
    """
    Destroy operator that removes customers based on potential cost improvement.
    Removes customers whose removal would lead to the largest cost reduction.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> Tuple[VRPData, List[List[int]], List[List[int]],float]:
        """
        Removes customers that would give the largest cost reduction when removed.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on cost improvement
            
        Returns:
            Tuple containing:
                - VRPData: Modified state with updated routes and unassigned customers
                - List[List[int]]: New edges added (source and target nodes)
                - List[List[int]]: Deleted edges (source and target nodes)
        """
        customers_to_remove = state.compute_worst_nodes()[:num_customers_to_remove].tolist()
        return self.remove_customers(state, customers_to_remove)
        

        

