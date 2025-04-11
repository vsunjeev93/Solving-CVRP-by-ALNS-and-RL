import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from copy import deepcopy # Import deepcopy

from actor import actor
from state_transition import state_transition
# Use data_generator to create a loader, even for a single instance
from graph_data import data_generator 
from train import move_graph_to_device # Use the device moving utility

def plot_routes(ax, vrp_data, title):
    """Plots the VRP routes on a given matplotlib axis."""
    points = vrp_data.points
    ax.plot(points[0, 0], points[0, 1], 'rs', label='Depot') # Depot
    ax.plot(points[1:, 0], points[1:, 1], 'bo', label='Customers') # Customers
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(vrp_data.routes))) # Different color for each route
    
    total_cost = 0
    for i, route in enumerate(vrp_data.routes):
        route_points = points[route, :]
        ax.plot(route_points[:, 0], route_points[:, 1], '-', color=colors[i], linewidth=1.5)
        
        # Calculate route cost
        route_cost = sum(vrp_data.distance_matrix[route[j], route[j+1]] for j in range(len(route)-1))
        total_cost += route_cost
        print(f"  Route {i+1}: Cost={route_cost:.4f}, Nodes={route}")
        
    # Add labels and title
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"{title} (Total Cost: {total_cost:.4f})")
    ax.legend()
    ax.grid(True)

def test(actor_checkpoint_path, num_customers=10, vehicle_capacity=4, budget=20, batch_size=8, seed=42):
    """Loads an actor model, runs ALNS on batched instances, and tracks cost history."""
    if not os.path.exists(actor_checkpoint_path):
        print(f"Error: Checkpoint file not found at {actor_checkpoint_path}")
        return
        
    # Set device
    device = torch.device('cpu') # Testing usually done on CPU
    print(f"Using device: {device}")
    
    # Set seed for instance generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Generating instances with seed: {seed}")
    
    # --- Load Model --- 
    actor_model = actor(7, 128, 2, 3, 9).to(device)
    actor_model.load_state_dict(torch.load(actor_checkpoint_path, map_location=device))
    actor_model.eval() # Set model to evaluation mode
    print(f"Loaded actor model from {actor_checkpoint_path}")
    
    # --- Generate DataLoader with batched instances --- 
    print(f"Generating VRP instances (batch_size={batch_size}, customers={num_customers}, capacity={vehicle_capacity})...")
    # Use data_generator to create batched instances
    test_loader = data_generator(num_customers, vehicle_capacity, instances=batch_size, batch_size=batch_size)
    
    # --- Extract the batch instance from the loader --- 
    graph = next(iter(test_loader))
    graph = move_graph_to_device(graph, device)
    
    # Save initial state for reference (just for the first instance in batch)
    initial_vrp_state = deepcopy(graph.state[0])
    initial_cost_calculated = 0 # Calculate initial cost manually for reference
    for route in initial_vrp_state.routes:
        for j in range(len(route)-1):
            initial_cost_calculated += initial_vrp_state.distance_matrix[route[j], route[j+1]]
    print(f"Initial solution calculated cost (first instance): {initial_cost_calculated:.4f}")
    
    # --- Run ALNS with Loaded Actor --- 
    print(f"\nRunning ALNS with loaded model for budget {budget} on {batch_size} instances...")
    state = graph 
    mask = torch.tensor([True]*5 + [False]*4).to(device)
    destroy = True
    n_iter = 0
    
    # Initialize batch best costs
    best_costs = graph.cost.clone()  # Start with initial costs
    best_state_datas = [deepcopy(s) for s in graph.state]  # Store best states

    # --- Define constant for number of customers to remove --- 
    num_to_remove_const = 5 
    print(f"Using fixed num_customers_to_remove = {num_to_remove_const}")
    
    # Track costs per iteration
    cost_history = []
    best_cost_history = [best_costs.mean().item()]  # Start with initial cost
    
    greedy=False
    with torch.no_grad(): 
        while True:
            current_mask = mask if destroy else ~mask
            action, _ = actor_model(state, current_mask, greedy=greedy)
            
            state, destroy = state_transition(state, action, num_to_remove_const, destroy)
            
            # Update best costs and states if current costs are better
            if destroy:
                # Update best costs for each instance in the batch if better
                for i in range(len(state.state)):
                    current_cost = state.cost[i].item()
                    if current_cost < best_costs[i].item():
                        best_costs[i] = state.cost[i]
                        best_state_datas[i] = deepcopy(state.state[i])
                
                # Track the best average cost at this iteration
                best_cost_history.append(best_costs.mean().item())
                
                # Print progress every few iterations
                if n_iter % 5 == 0:
                    print(f"  Iter {n_iter}: Avg best cost = {best_costs.mean().item():.4f}")
                
            n_iter += 1
            # if n_iter>50:
            #     greedy=False
            
            # Termination condition
            if n_iter >= budget and destroy:
                print(f"Termination condition met: Budget {budget} reached after iteration {n_iter}.")
                break

    # Calculate and print final statistics
    avg_best_cost = best_costs.mean().item()
    min_best_cost = best_costs.min().item()
    max_best_cost = best_costs.max().item()
    std_best_cost = best_costs.std().item()
    
    print(f"\nALNS finished.")
    print(f"Average best cost: {avg_best_cost:.4f}")
    print(f"Min best cost: {min_best_cost:.4f}")
    print(f"Max best cost: {max_best_cost:.4f}")
    print(f"Std dev of best costs: {std_best_cost:.4f}")
    
    # Return the cost history
    return best_cost_history

def run_multiple_seeds(actor_checkpoint_path, num_customers=10, vehicle_capacity=4, budget=20, 
                      batch_size=8, seeds=None, num_seeds=5):
    """
    Run the test function with multiple seeds and track the iteration history.
    
    Args:
        actor_checkpoint_path: Path to the actor model checkpoint
        num_customers: Number of customers in VRP instances
        vehicle_capacity: Vehicle capacity for VRP instances
        budget: Number of ALNS iterations
        batch_size: Number of instances to test in parallel
        seeds: List of seeds to use (if None, generates seeds)
        num_seeds: Number of seeds to generate if seeds is None
    """
    if seeds is None:
        # Generate seeds if not provided
        base_seed = 42
        seeds = [base_seed + i*100 for i in range(num_seeds)]
    
    print(f"Running with {len(seeds)} seeds: {seeds}")
    
    # Store iteration histories for each seed
    all_histories = []
    
    # Run test for each seed
    for i, seed in enumerate(seeds):
        print(f"\nRunning seed {i+1}/{len(seeds)} (seed={seed})...")
        history = test(
            actor_checkpoint_path=actor_checkpoint_path,
            num_customers=num_customers,
            vehicle_capacity=vehicle_capacity,
            budget=budget,
            batch_size=batch_size,
            seed=seed
        )
        all_histories.append(history)
        print(f"Seed {seed} final best avg cost: {history[-1]:.4f}")
    
    # Find the maximum length of histories
    max_length = max(len(h) for h in all_histories)
    
    # Pad shorter histories by repeating their last value
    padded_histories = []
    for history in all_histories:
        if len(history) < max_length:
            padded = history + [history[-1]] * (max_length - len(history))
        else:
            padded = history
        padded_histories.append(padded)
    
    # Convert to numpy array for easier computation
    histories_array = np.array(padded_histories)
    
    # Calculate statistics across seeds
    mean_history = np.mean(histories_array, axis=0)
    
    # Plot only the average best cost vs iteration
    plt.figure(figsize=(10, 6))
    iterations = np.arange(max_length)
    
    # Plot mean with thick line
    plt.plot(iterations, mean_history, 'b-', linewidth=2.5, label='Mean Best Cost')
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Best Cost')
    checkpoint_name = os.path.splitext(os.path.basename(actor_checkpoint_path))[0]
    plt.title(f'Average Best Cost vs Iteration (Across {len(seeds)} Seeds)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot to file
    plot_filename = f"cost_vs_iteration_c{num_customers}_v{vehicle_capacity}_seeds{len(seeds)}_{checkpoint_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved iteration plot to {plot_filename}")
    
    plt.show()
    
    # Print final statistics
    print("\nFinal Statistics (across all seeds):")
    print(f"Average final best cost: {mean_history[-1]:.4f}")
    print(f"Best seed: {seeds[np.argmin([h[-1] for h in all_histories])]} with cost {np.min([h[-1] for h in all_histories]):.4f}")
    print(f"Worst seed: {seeds[np.argmax([h[-1] for h in all_histories])]} with cost {np.max([h[-1] for h in all_histories]):.4f}")
    
    return mean_history, padded_histories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained VRP ALNS model on batched instances.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="./checkpoints/actor_c20_v30_final.pth", 
        help="Path to the actor model checkpoint file (.pth)"
    )
    parser.add_argument(
        "--customers", 
        type=int, 
        default=20, # Use a slightly larger default for testing 
        help="Number of customers in the test instance"
    )
    parser.add_argument(
        "--capacity", 
        type=int, 
        default=30, # Example capacity
        help="Vehicle capacity for the test instance"
    )
    parser.add_argument(
        "--budget", 
        type=int, 
        default=2000, # Example budget for testing
        help="ALNS iteration budget for testing"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=20, # Default batch size
        help="Number of instances to test in parallel"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123, # Use a different seed for testing than training
        help="Random seed for generating the test instance"
    )
    parser.add_argument(
        "--multi_seed", 
        action="store_true",
        help="Run with multiple seeds and plot average cost vs iteration"
    )
    parser.add_argument(
        "--num_seeds", 
        type=int, 
        default=5,
        help="Number of different seeds to run when using multi_seed"
    )
    
    args = parser.parse_args()
    
    if args.multi_seed:
        # Run with multiple seeds and plot average cost vs iteration
        run_multiple_seeds(
            actor_checkpoint_path=args.checkpoint,
            num_customers=args.customers,
            vehicle_capacity=args.capacity,
            budget=args.budget,
            batch_size=args.batch_size,
            num_seeds=args.num_seeds
        )
    else:
        # Run with a single seed and still show cost vs iteration plot
        history = test(
            actor_checkpoint_path=args.checkpoint, 
            num_customers=args.customers, 
            vehicle_capacity=args.capacity, 
            budget=args.budget,
            batch_size=args.batch_size,
            seed=args.seed
        )
        
        # Plot cost vs iteration for single seed
        plt.figure(figsize=(10, 6))
        iterations = np.arange(len(history))
        
        plt.plot(iterations, history, 'b-', linewidth=2.5, label='Best Cost')
        
        plt.xlabel('Iteration')
        plt.ylabel('Average Best Cost')
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        plt.title(f'Best Cost vs Iteration (Seed {args.seed})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save plot to file
        plot_filename = f"cost_vs_iteration_c{args.customers}_v{args.capacity}_seed{args.seed}_{checkpoint_name}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved iteration plot to {plot_filename}")
        
        plt.show()
