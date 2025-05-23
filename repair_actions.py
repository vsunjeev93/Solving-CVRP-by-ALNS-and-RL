"""
Repair operators for VRP using Adaptive Large Neighborhood Search (ALNS).
Reinserts customers back into routes after removal by destroy operators.
"""

from vrp_data import VRPData
import numpy as np
from typing import List

class RepairUtils:
    """Utility class with insertion cost calculations and position finding."""
    
    def insertion_cost(self, state: VRPData, node: int, node_prev: int, node_after: int) -> float:
        """
        Calculate cost of inserting node between two nodes.
        
        Args:
            state: VRPData with distance matrix
            node, node_prev, node_after: Nodes involved in insertion
            
        Returns:
            Cost difference after insertion
        """
        current_cost = state.distance_matrix[node_prev, node_after]
        new_cost = state.distance_matrix[node_prev, node] + state.distance_matrix[node, node_after]
        return new_cost - current_cost
    
    def get_min_insertion_cost(self, state: VRPData, node: int) -> tuple[float, int, int, List]:
        """
        Find minimum cost insertion position across all routes.
        
        Args:
            state: Current VRP state
            node: Node to insert
            
        Returns:
            (min_cost, route_id, position_index, all_candidates)
        """
        min_cost = float('inf')
        route_id = len(state.routes)  # Default to new route
        index = 1  # Default position
        candidates_cost_info = []  # (cost, route_id, index) tuples
        
        # Try existing routes
        for i, route in enumerate(state.routes):
            total_demand_in_route = sum(state.node_demand[n] for n in route)
            if total_demand_in_route + state.node_demand[node] <= state.vehicle_capacity:
                for node_index in range(1, len(route)):
                    prev_node = route[node_index-1]
                    next_node = route[node_index]
                    if node==prev_node or node==next_node:
                        raise Exception(node,prev_node,next_node)
                    cost = self.insertion_cost(state, node, prev_node, next_node)
                    candidates_cost_info.append((cost, i, node_index))
                    if cost < min_cost:
                        min_cost = cost
                        route_id, index = i, node_index
        
        # New route cost
        if route_id == len(state.routes):
            min_cost = 2 * state.distance_matrix[0, node]
            candidates_cost_info.append((min_cost, 1, route_id))
            
        return min_cost, route_id, index, candidates_cost_info

class GreedyRepair(RepairUtils):
    """Inserts customers at position with minimum insertion cost."""
    
    def insert_customers(self, state: VRPData) -> tuple[VRPData,tuple[List[int]],tuple[List[int]],float]:
        """
        Insert all customers using greedy approach.
        
        Takes first unassigned customer, finds best position, inserts.
        Creates new route if needed.
        
        Args:
            state: Current VRP state
            
        Returns:
            Updated state with all customers assigned
        """
        cost_change=0
        while len(state.unassigned_customers) > 0:
            current_customer = state.unassigned_customers[0]
            # print(state.unassigned_customers,state.routes)
            _, route_id, index, _ = self.get_min_insertion_cost(state, current_customer)
            
            if route_id == len(state.routes):
                # Create new route
                state.routes.append([0, current_customer, 0])
                cost_change+=state.distance_matrix[0,current_customer]*2
            else:
                # Insert at best position
                cost_change+=state.distance_matrix[current_customer,state.routes[route_id][index-1]]+state.distance_matrix[current_customer,state.routes[route_id][index]]
                cost_change-=state.distance_matrix[state.routes[route_id][index],state.routes[route_id][index-1]]
                if state.distance_matrix[current_customer,state.routes[route_id][index-1]]+state.distance_matrix[current_customer,state.routes[route_id][index]]==float('inf') or state.distance_matrix[current_customer,state.routes[route_id][index-1]]+state.distance_matrix[current_customer,state.routes[route_id][index]] == -float('inf'):
                    raise Exception(f'cost is inf {state.routes[route_id][index-1]} {state.routes[route_id][index]} {current_customer} {state.routes[route_id]} ',state.distance_matrix)
                state.routes[route_id] = state.routes[route_id][:index] + [current_customer] + state.routes[route_id][index:]
                
                
            state.unassigned_customers.pop(0)
        
        return state,cost_change


class SortedGreedyRepair(GreedyRepair):
    """Sorts customers by insertion cost before inserting."""
    
    def insert_customers(self, state: VRPData) -> tuple[VRPData,tuple[List[int]],tuple[List[int]],float]:
        """
        Sort customers by insertion cost, then insert.
        
        Prioritizes "easier" customers first (lower insertion costs).
        
        Args:
            state: Current VRP state
            
        Returns:
            Updated state with all customers assigned
        """
        # Calculate costs for all customers
        insertion_dict = {node: self.get_min_insertion_cost(state, node) for node in state.unassigned_customers}
        
        # Sort by cost (ascending)
        state.unassigned_customers = sorted(state.unassigned_customers, key=lambda x: insertion_dict[x][0])
        
        # Insert using parent method
        return super().insert_customers(state)

class RegretkRepair(GreedyRepair):
    """
    Prioritizes customers by regret value (k-th best minus best insertion cost).
    Higher regret customers inserted first to avoid future penalties.
    """
    
    def __init__(self, k=2):
        """Initialize with regret parameter k."""
        self.k = k
        
    def insert_customers(self, state: VRPData) -> tuple[VRPData,tuple[List[int]],tuple[List[int]],float]:
        """
        Calculate regret scores, sort customers (highest first), then insert.
        
        Args:
            state: Current VRP state
            
        Returns:
            Updated state with all customers assigned
            
        """
        regret_scores = {}
        for cust in state.unassigned_customers:
            _, route_id, index, candidate_cost_info = self.get_min_insertion_cost(state, cust)
            candidate_cost_info = sorted(candidate_cost_info, key=lambda x: x[0])
            if len(candidate_cost_info) == 0:
                # If no positions available, this is an error in problem setup
                raise Exception('No feasible insertion positions found for customer')
            elif len(candidate_cost_info) == 1:
                # If only one position available, regret is 0
                regret_score = 0
            else:
                # Use min(k, number of positions) to calculate regret
                k_to_use = min(self.k, len(candidate_cost_info))
                regret_score = candidate_cost_info[k_to_use-1][0] - candidate_cost_info[0][0]
            regret_scores[cust] = regret_score
            
        # Sort by regret (descending)
        state.unassigned_customers = sorted(state.unassigned_customers, key=lambda x: regret_scores[x], reverse=True)
        
        return super().insert_customers(state)



    

                
