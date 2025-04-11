import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import index_to_mask
from vrp_data import VRPData


"""Graph utilities for VRP instances using PyTorch Geometric."""

def generate_graph_and_initial_solution(customers, vehicle_capacity):
    """
    Generate a graph representation of a VRP instance with initial solution.
    
    Args:
        customers (int): Number of customers
        vehicle_capacity (float): Maximum vehicle capacity
        
    Returns:
        Data: Graph with node features, edges, center node, masks, and solution state
    
    Raises:
        Exception: If vehicle capacity is insufficient
    """
    points = torch.rand((customers+1, 2)) #including depot
    demand = torch.randint(1,10,(customers,)).float()  #(customers,1)
    if vehicle_capacity<torch.max(demand):
        raise Exception('vehicle capacity is less than demand of a some customers')

    demand = torch.cat([torch.tensor([0]), demand], dim=0)  # (customers+1,1)
    center = torch.mean(points, dim=0)
    distance = torch.linalg.norm(points - center, dim=1)
    angle_center = torch.atan2(points[:, 1] - center[1], points[:, 0] - center[0])
    features = []
    edges = []
    for i in range(customers+1):
        if i==0:
            is_cust=0
        else:
            is_cust=1
        features.append(torch.tensor([
            # is_cust, #depot=0, cust=1, center=-1
            -1, # placeholder for  position in route node is in -1 for depot and center node
            -1, # placeholder for route node is in -1 for depot and center node
            demand[i], #demand for the node
            points[i][0], #x coord
            points[i][1], # y coord
            distance[i], # distance between center node and cust/depot
            angle_center[i], #angle between center node and cust/depot
       ]))
        # edges.append(torch.tensor([i,customers+1]))# adding edge from customer/depot to center node
        edges.append(torch.tensor([customers+1,i]))
        edges.append(torch.tensor([i,customers+1]))
    features.append(torch.tensor([-1,-1,0,center[0], center[1],0,0]))
    features = torch.stack(features)
    # get initial solution
    routes, total_cost = initial_solution(points,demand,vehicle_capacity)
    vrp_state=VRPData(points=points.numpy(),vehicle_capacity=vehicle_capacity,node_demand=demand.numpy(),routes=routes)
    # add edges in routes:
    for i,route in enumerate(vrp_state.routes):
        for node_index,node in enumerate(route[:-1]):
            if node!=0:
                features[node,0]=node_index
                features[node,1]=i
            edges.append(torch.tensor([node,route[node_index+1]]))
            edges.append(torch.tensor([route[node_index+1],node]))
    
    edge_index = torch.stack(edges)
    mask = index_to_mask(torch.tensor(range(1,customers)), size=customers + 2)#masks depot and center node
    # print(total_cost)
    graph = Data(
        x=features,
        edge_index=edge_index.t().contiguous(),
        center_node_index=torch.tensor([customers+1]),
        mask=mask,# masks depot and center node
        graph_id_index=torch.tensor([0]),
        state=vrp_state,
        cost=torch.tensor(total_cost)
    )
    return graph

def data_generator(n, vehicle_capacity, instances=10000, batch_size=12):
    """
    Generate a DataLoader with multiple VRP instances.
    
    Args:
        n (int): Customers per instance
        vehicle_capacity (float): Maximum capacity
        instances (int, optional): Number of instances. Default: 10000
        batch_size (int, optional): Batch size. Default: 12
        
    Returns:
        DataLoader: Batched graph instances
    """
    graphs = []
    # torch.manual_seed(5)
    for _ in range(instances):
        graph = generate_graph_and_initial_solution(n, vehicle_capacity)
        # Ensure all tensors are on CPU - we don't need to do this as they're already on CPU
        graphs.append(graph)
    loader = DataLoader(graphs, batch_size=batch_size)
    return loader

def initial_solution(points, demand, vehicle_capacity):
    """
    Generate initial VRP solution using nearest neighbor heuristic.
    
    Args:
        points (Tensor): Node coordinates (n+1, 2)
        demand (Tensor): Node demands (n+1)
        vehicle_capacity (float): Maximum capacity
        
    Returns:
        tuple: (routes, total_cost) where routes is a list of routes starting/ending at depot (0)
               and total_cost is the sum of all route distances
    """
    n = len(points) - 1  # number of customers (excluding depot)
    unvisited_mask = torch.ones(n + 1, dtype=torch.bool)  # Track unvisited customers
    unvisited_mask[0] = False  # Depot is not a customer
    routes = []
    current_route = [0]  # start with depot (index 0)
    current_load = 0
    current_pos = 0  # depot position
    total_cost = 0.0
    
    while unvisited_mask.any():
        # Calculate distances to all points at once
        distances = torch.norm(points - points[current_pos], dim=1)
        
        # Create feasibility mask combining unvisited and capacity constraints
        capacity_mask = (current_load + demand.squeeze()) <= vehicle_capacity
        feasible_mask = unvisited_mask & capacity_mask
        
        # Set distances of non-feasible points to infinity
        distances[~feasible_mask] = float('inf')
        
        # Find nearest feasible customer
        nearest = torch.argmin(distances).item()
        
        if distances[nearest] == float('inf'):  # No feasible customer found
            # Add return to depot distance
            total_cost += torch.norm(points[current_pos] - points[0]).item()
            current_route.append(0)
            routes.append(current_route)
            # Start a new route
            current_route = [0]
            current_load = 0
            current_pos = 0
            continue
            
        # Add nearest customer to route
        current_route.append(nearest)
        current_load += demand[nearest].item()
        # Add distance to total cost
        total_cost += distances[nearest].item()
        current_pos = nearest
        unvisited_mask[nearest] = False
    
    # Add return to depot distance for last route
    total_cost += torch.norm(points[current_pos] - points[0]).item()
    current_route.append(0)
    routes.append(current_route)
    
    return routes, total_cost

if __name__=='__main__':
    graphs=data_generator(40,10,2,2)
    torch.manual_seed(42)
    for graph in graphs:
        # print(graph.state)
        for state in graph.state:
            print(state.cost,state.routes)
        print(graph.edge_index)
    # print(graph['x'])