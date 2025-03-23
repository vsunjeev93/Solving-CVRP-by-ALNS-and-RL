from destroy_actions import RandomRemoval, DemandRelatedRemoval, GeographicRelatedRemoval, RouteRelatedRemoval, GreedyRemoval
from repair_actions import GreedyRepair, SortedGreedyRepair,  RegretkRepair
from torch_geometric.data import Data
from vrp_data import VRPData
from torch_geometric.utils import index_to_mask, coalesce
import torch
from typing import List, Tuple
def state_transition(graph: Data, action: List[int], num_customers_to_remove: int, destroy: bool = True) -> Tuple[Data, bool]:
    """
    Apply destroy or repair operators to VRP instances based on current state.
    
    This function handles the state transition in the Adaptive Large Neighborhood Search (ALNS)
    algorithm by applying either destroy or repair operators to VRP solutions. It manages both
    the logical state of the VRP solution and the corresponding graph structure.
    
    Args:
        graph (Data): PyTorch Geometric Data object containing VRP instances.
               Must have a 'state' attribute holding VRPData objects and a 'graph_id_index'
               attribute to track instance offsets in batched graphs.
        action (List[int]): Indices of operators to apply.
               For destroy phase (destroy=True): Use indices 0-4 for destroy operators.
               For repair phase (destroy=False): Use indices 5-8 for repair operators.
        num_customers_to_remove (int): Number of customers to remove if in destroy phase.
               Ignored in repair phase (can be None).
        destroy (bool, optional): Flag indicating current phase.
               True for destroy phase, False for repair phase. Defaults to True.
    
    Returns:
        tuple[Data, bool]: 
            - Modified graph with updated state, including modified routes, unassigned customers,
              and updated edge structure
            - Updated destroy flag (toggled from input value) to track the next required phase
              
    Raises:
        Exception: If a repair operator (index 5-8) is selected during destroy phase
                  or if a destroy operator (index 0-4) is selected during repair phase.
    
    Note:
        The destroy flag is toggled after each operation to alternate between destroy
        and repair phases. This ensures the proper sequence of operations in the ALNS algorithm.
    """
    destroy_list=[RandomRemoval(), DemandRelatedRemoval(), GeographicRelatedRemoval(), RouteRelatedRemoval(), GreedyRemoval()]
    repair_list=[GreedyRepair(),SortedGreedyRepair(),RegretkRepair(2),RegretkRepair(3)]
    action_list=destroy_list+repair_list
    if destroy:
        new_states=[]
        for i,state in enumerate(graph.state):
            if action[i]>=len(destroy_list):
                raise Exception('A destory operator has not been chosen')
            destroy_operator=action_list[action[i]]
            new_state,new_edges,delete_edges=destroy_operator.action(state,num_customers_to_remove)
            new_states.append(new_state)
            graph_index=graph.graph_id_index[i]
            graph=remove_edges(graph,delete_edges,graph_index)
            graph=add_edges(graph,new_edges,graph_index)
        graph.state=new_states
        destroy=not destroy
    else:
        new_states=[]
        for i,state in enumerate(graph.state):
            if action[i]<len(destroy_list):
                raise Exception('A repair operator has not been chosen')
            repair_operator=action_list[action[i]]
            new_state,new_edges,delete_edges=repair_operator.insert_customers(state)
            new_states.append(new_state)
            graph_index=graph.graph_id_index[i]
            graph=remove_edges(graph,delete_edges,graph_index)
            graph=add_edges(graph,new_edges,graph_index)
        graph.state=new_states
        destroy=not destroy
    return graph, destroy
def remove_edges(graph: Data, delete_edges: List[List[int]], graph_index: int) -> Data:
    """
    Remove edges from the graph during state transition.
    
    Args:
        graph (Data): PyTorch Geometric Data object containing the graph.
        delete_edges (List[List[int]]): List of edges to remove in format [sources, targets].
        graph_index (int): Offset to add to node indices for batched graphs.
        
    Returns:
        Data: Graph with specified edges removed.
        
    Note:
        This function uses coalesce to efficiently handle edge removal by setting
        weights and then filtering based on those weights.
    """
    sources_to_remove=[graph_index+node for node in delete_edges[0]]
    targets_to_remove=[graph_index+node for node in delete_edges[1]]
    edge_index_remove=torch.tensor([sources_to_remove,targets_to_remove])
    all_edge_index=torch.cat([graph.edge_index,edge_index_remove],dim=1)
    edge_weights=torch.cat([torch.zeros(graph.edge_index.size(1)),torch.ones(edge_index_remove.size(1))])
    all_edge_index,edge_weights=coalesce(all_edge_index,edge_weights)
    edge_index_after_removal=all_edge_index[:,edge_weights==0]
    graph.edge_index=edge_index_after_removal
    return graph

def add_edges(graph: Data, new_edges: List[List[int]], graph_index: int) -> Data:
    """
    Add new edges to the graph during state transition.
    
    Args:
        graph (Data): PyTorch Geometric Data object containing the graph.
        new_edges (List[List[int]]): List of edges to add in format [sources, targets].
        graph_index (int): Offset to add to node indices for batched graphs.
        
    Returns:
        Data: Graph with new edges added.
    """
    #new edges
    sources_to_add=[graph_index+node for node in new_edges[0]]
    targets_to_add=[graph_index+node for node in new_edges[1]]
    new_edges=torch.tensor([sources_to_add,targets_to_add])

    #get new set of edges
    graph.edge_index=torch.cat([graph.edge_index,new_edges],dim=1)
    return graph
if __name__=='__main__':
    # Basic usage example of state transition in a destroy-repair cycle
    from graph_data import generate_graph_and_initial_solution
    
    # Create a VRP instance with 10 customers and vehicle capacity of 5
    graph = generate_graph_and_initial_solution(10, 5)
    
    # Print initial solution
    print("Initial solution:")
    print(f"Routes: {graph.state[0].routes}")
    print(f"Cost: {graph.state[0].cost}")
    
    # Apply destroy operator (GeographicRelatedRemoval)
    print("\nApplying destroy operator (Geographic Related Removal)...")
    destroy_action = [2]  # Index 2 corresponds to GeographicRelatedRemoval
    graph, destroy_flag = state_transition(graph, destroy_action, num_customers_to_remove=2, destroy=True)
    
    # Print solution after destroy
    print(f"Routes after destroy: {graph.state[0].routes}")
    print(f"Unassigned customers: {graph.state[0].unassigned_customers}")
    print(f"Destroy flag: {destroy_flag}")
    
    # Apply repair operator (RegretkRepair with k=3)
    print("\nApplying repair operator (Regretk Repair with k=3)...")
    repair_action = [7]  # Index 7 corresponds to RegretkRepair(3)
    graph, destroy_flag = state_transition(graph, repair_action, num_customers_to_remove=None, destroy=destroy_flag)
    
    # Print final solution
    print(f"Routes after repair: {graph.state[0].routes}")
    print(f"Final cost: {graph.state[0].update_cost()}")
    print(f"Destroy flag: {destroy_flag}")
    print(f"Unassigned customers: {graph.state[0].unassigned_customers}")


