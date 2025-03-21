from copy import deepcopy
from typing import List
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
    def remove_customers(self, routes: List[List[int]], customers_to_remove: List[int]) -> List[List[int]]:
        """
        Removes specified customers from the given routes.
        
        Args:
            routes: List[List[int]] - List of routes where each route is a list of customer indices
            customers_to_remove: List[int] - List of customer indices to remove from routes
            
        Returns:
            List[List[int]] - Modified routes with specified customers removed
        """
        new_routes = []
        for route in routes:
            new_route = [node for node in route if node not in customers_to_remove]
            if len(new_route) > 2:  # Keep route if it has more than just depot-depot
                new_routes.append(new_route)
        return new_routes

class RandomRemoval(Destroy):
    """
    Random removal destroy operator that randomly selects and removes customers from routes.
    Inherits from the base Destroy class. Updates the VRPData state with removed customers
    for tracking unassigned customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> VRPData:
        """
        Randomly removes a specified number of customers from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to randomly remove
            
        Returns:
            VRPData - Modified state with updated routes and unassigned customers
            
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
        new_routes = self.remove_customers(state.routes, customers_to_remove)
        state.routes = new_routes
        state.unassigned_customers = customers_to_remove
        return state

class DemandRelatedRemoval(Destroy):
    """
    Destroy operator that removes customers with similar demands.
    Selects a random customer and removes its nearest neighbors based on demand similarity.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> VRPData:
        """
        Removes customers with similar demands from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on demand similarity
            
        Returns:
            VRPData - Modified state with updated routes and unassigned customers
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
        
        new_routes = self.remove_customers(state.routes, customers_to_remove)
        state.routes = new_routes
        state.unassigned_customers = customers_to_remove
        return state

class GeographicRelatedRemoval(Destroy):
    """
    Destroy operator that removes geographically close customers.
    Selects a random customer and removes its nearest neighbors based on Euclidean distance.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> VRPData:
        """
        Removes geographically close customers from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on geographic proximity
            
        Returns:
            VRPData - Modified state with updated routes and unassigned customers
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
        
        new_routes = self.remove_customers(state.routes, customers_to_remove)
        state.routes = new_routes
        state.unassigned_customers = customers_to_remove
        return state

class RouteRelatedRemoval(Destroy):
    """
    Destroy operator that removes customers that are close to each other in route sequence.
    Selects a random customer and removes its nearest neighbors based on route position.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> VRPData:
        """
        Removes customers that are close in route sequence from the current solution.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on route proximity
            
        Returns:
            VRPData - Modified state with updated routes and unassigned customers
        """
        first_customer = random.choice(state.customers)
        customers_to_remove = [first_customer]
        
        # Get nearest neighbors by route sequence, excluding those already selected
        neighbors = state.nearest_neighbors_by_route[first_customer].tolist()
        for neighbor in neighbors:
            if len(customers_to_remove) >= num_customers_to_remove:
                break
            if neighbor != 0 and neighbor not in customers_to_remove:  # Skip depot and already selected
                customers_to_remove.append(neighbor)
        
        new_routes = self.remove_customers(state.routes, customers_to_remove)
        state.routes = new_routes
        state.unassigned_customers = customers_to_remove
        return state

class GreedyRemoval(Destroy):
    """
    Destroy operator that removes customers based on potential cost improvement.
    Removes customers whose removal would lead to the largest cost reduction.
    Updates the VRPData state with removed customers.
    """
    def action(self, state: VRPData, num_customers_to_remove: int) -> VRPData:
        """
        Removes customers that would give the largest cost reduction when removed.
        
        Args:
            state: VRPData - Current state of the VRP instance containing routes and other data
            num_customers_to_remove: int - Number of customers to remove based on cost improvement
            
        Returns:
            VRPData - Modified state with updated routes and unassigned customers
        """
        customers_to_remove = state.compute_worst_nodes()[:num_customers_to_remove]
        new_routes = self.remove_customers(state.routes, customers_to_remove)
        state.routes = new_routes
        state.unassigned_customers = customers_to_remove
        return state

        

