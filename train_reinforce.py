import torch
from graph_data import data_generator
from actor import actor
from critic import critic
from state_transition import state_transition
import torch.nn.functional as F
import time # Add time import
import numpy as np  # Add numpy for statistics
import os # Add os for path operations
import yaml # Add yaml import

def compute_losses(rewards, log_actions, state_values):
    actor_losses = []
    critic_losses = []
    cum_rewards=torch.cumsum(rewards,dim=-1)
    # print(cum_rewards,'cum_rewards')
    for index, (log_action, state_value) in enumerate(zip(log_actions, state_values)):
        # rewards_to_go = torch.stack(rewards[index:], dim=-1).sum(dim=-1)
        rewards_to_go=cum_rewards[:,index]
        # print(rewards_to_go.size(),cum_rewards.size(),state_value.size(),log_action.size())
        # assert 1==2
        # print(rewards_to_go,'reward')
        actor_loss = (rewards_to_go - state_value.squeeze()).detach() * log_action
        critic_loss = F.smooth_l1_loss(rewards_to_go, state_value.squeeze())
        # print('here')
        # print(rewards_to_go - state_value.squeeze(),log_action,actor_loss,critic_loss.size(),critic_loss)
        actor_losses.append(actor_loss.mean())
        critic_losses.append(critic_loss)
    # print(actor_losses)
    actor_loss = torch.stack(actor_losses).sum()
    critic_loss = torch.stack(critic_losses).sum()
    # print(actor_loss.size(),critic_loss.size())
    # assert 1==2
    
    return actor_loss, critic_loss

def move_graph_to_device(graph, device):
    """Move all tensor attributes of a graph to the specified device."""
    for key in graph.keys():  # keys() is a method, not a property
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].to(device)
    return graph

# Helper function to compute gradient statistics
def compute_grad_stats(model):
    total_norm = 0
    max_norm = 0
    min_norm = float('inf')
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_norm = max(max_norm, param_norm.item())
            min_norm = min(min_norm, param_norm.item())
            count += 1
    total_norm = total_norm ** 0.5
    avg_norm = total_norm / count if count > 0 else 0
    return {
        'avg': avg_norm,
        'max': max_norm,
        'min': min_norm if min_norm != float('inf') else 0,
        'total': total_norm
    }

def train(collect_stats=True, seed=42):
    # Set device
    device = torch.device('cpu')  # Use CPU as MPS has compatibility issues with PyTorch Geometric
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Using random seed: {seed}")
    
    # Checkpoint directory
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize models
    mask = torch.tensor([True]*5 + [False]*4).to(device)
    actor_model = actor(7, 128, 2, 3, 9).to(device)
    critic_model = critic(7, 128, 2, 3).to(device)
    # for param in actor_model.parameters():
    #     if len(param.shape) > 1:
    #         torch.nn.init.kaiming_normal_(param)
    # for param in critic_model.parameters():
    #     if len(param.shape) > 1:
    #         torch.nn.init.kaiming_normal_(param)
    lr=10**-5
    optimizer = torch.optim.Adam(list(actor_model.parameters()) + list(critic_model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    epochs=20
    
    # Problem parameters
    num_customers = 100
    vehicle_capacity = 50
    
    # Timing variables
    total_batch_creation_time = 0
    total_training_time = 0
    total_batches_processed = 0
    
    # Logging variables - only initialize if collecting stats
    if collect_stats:
        epoch_rewards = []
        epoch_actor_grads = []
        epoch_critic_grads = []
        epoch_value_errors = []
        epoch_best_costs = []  # Add list to track best costs per epoch
        # Initialize cumulative action counters
        destroy_action_counts = torch.zeros(5, device=device)  # 5 destroy operators (0-4)
        repair_action_counts = torch.zeros(4, device=device)   # 4 repair operators (5-8)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Time data generation
        data_gen_start = time.time()
        batches = data_generator(num_customers, vehicle_capacity, instances=6*20, batch_size=6)
        data_gen_time = time.time() - data_gen_start
        total_batch_creation_time += data_gen_time
        print(f"Data generation time: {data_gen_time:.2f}s")
        
        epoch_training_time = 0
        epoch_batches = 0
        
        # Per-epoch logging containers - only initialize if collecting stats
        if collect_stats:
            batch_rewards = []
            batch_actor_grads = []
            batch_critic_grads = []
            batch_value_errors = []
            batch_best_costs = []  # Add list to track best costs per batch
        
        for state in batches:
            batch_start_time = time.time()
            total_batches_processed += 1
            epoch_batches += 1
            
            # Ensure state is on the correct device
            state = move_graph_to_device(state, device)
            best_cost = state.cost
            print('initial cost',best_cost.mean())
            batch_size = len(state.state)
            log_actions = []
            rewards = []
            budget = 50
            num_customers_to_remove = 5
            destroy = True
            n_iter = 0
            state_values = []
            # state_value = critic_model(state)
            
            
            while True:
                action, log_action = actor_model(state, mask)
                log_actions.append(log_action)
                state_value = critic_model(state)
                state_values.append(state_value)
                
                # Track action distribution - only if collecting stats
                if collect_stats:
                    # Record actions in appropriate counter based on current phase
                    if destroy:
                        # In destroy phase, actions should be 0-4
                        for a in action:
                            if a < 5:  # Destroy operator
                                destroy_action_counts[a] += 1
                    else:
                        # In repair phase, actions should be 5-8
                        for a in action:
                            if a >= 5:  # Repair operator
                                repair_action_counts[a-5] += 1  # Adjust index
                
                state, destroy = state_transition(state, action, num_customers_to_remove, destroy)
                mask = ~mask
                n_iter += 1
                
                if n_iter >= budget and destroy:
                    lls=torch.stack(log_actions,dim=1)
                    ll_sum=lls.sum(dim=-1)
                    # reward=initial_cost-best_cost
                    reward=torch.maximum(best_cost-state.cost,torch.zeros(batch_size, device=device))
                    rewards.append(reward)
                    # print(reward.size())
                    rewards=torch.stack(rewards,dim=-1)
                    # print(best_cost.size(),state.cost.size(),rewards.size(),'mean rewards')
                    # assert 1==2
                    
                    # Log statistics - only if collecting stats
                    if collect_stats:
                        # Log reward
                        reward_val = reward.mean().item()
                        batch_rewards.append(reward_val)
                        
                        # Log best cost
                        best_cost_val = best_cost.mean().item()
                        batch_best_costs.append(best_cost_val)
                        
                        # Log value prediction error 
                        value_error = (reward - state_value).abs().mean().item()
                        batch_value_errors.append(value_error)
                    
                    # reward = reward.unsqueeze(-1)
                    # actor_loss=((reward.detach()-state_value.detach())*ll_sum).mean()
                    # critic_loss=F.smooth_l1_loss(reward.detach(), state_value)
                    # critic_loss=critic_loss.mean()
                    actor_loss,critic_loss=compute_losses(rewards,log_actions,state_values)
                    total_loss=actor_loss+critic_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critic_model.parameters(), max_norm=5, norm_type=2
                    )
                    torch.nn.utils.clip_grad_norm_(
                        actor_model.parameters(), max_norm=5, norm_type=2
                    )
                    optimizer.step()
                    # Log gradient statistics - only if collecting stats
                    if collect_stats:
                        actor_grad_stats = compute_grad_stats(actor_model)
                        batch_actor_grads.append(actor_grad_stats)
                    
                    
                    # Log gradient statistics - only if collecting stats
                    if collect_stats:
                        critic_grad_stats = compute_grad_stats(critic_model)
                        batch_critic_grads.append(critic_grad_stats)
                    
                    
                    # Log normalized action distribution - only if collecting stats
                    
                    break
                elif destroy:# this would mean unassigned customers =0 since last operator was repair
                    reward=torch.maximum(best_cost-state.cost,torch.zeros(batch_size, device=device))
                    # print(reward.size(),'here')
                    rewards.append(reward)
                    best_cost=torch.minimum(state.cost,best_cost)
                else:
                    rewards.append(torch.zeros((batch_size,), device=device))
            
            batch_time = time.time() - batch_start_time
            epoch_training_time += batch_time
            if epoch_batches % 5 == 0: # Print timing every 5 batches
                 print(f"Batch {epoch_batches} training time: {batch_time:.2f}s")
        
        # Calculate and print epoch statistics - only if collecting stats
        if collect_stats:
            # Calculate epoch statistics
            avg_return = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            epoch_rewards.append(avg_return)
            
            # Calculate average best cost
            avg_best_cost = sum(batch_best_costs) / len(batch_best_costs) if batch_best_costs else 0
            epoch_best_costs.append(avg_best_cost)
            
            # Calculate average actor gradient magnitude
            avg_actor_grad = sum(g['avg'] for g in batch_actor_grads) / len(batch_actor_grads) if batch_actor_grads else 0
            epoch_actor_grads.append(avg_actor_grad)
            
            # Calculate average critic gradient magnitude
            avg_critic_grad = sum(g['avg'] for g in batch_critic_grads) / len(batch_critic_grads) if batch_critic_grads else 0
            epoch_critic_grads.append(avg_critic_grad)
            
            # Calculate average value prediction error
            avg_value_error = sum(batch_value_errors) / len(batch_value_errors) if batch_value_errors else 0
            epoch_value_errors.append(avg_value_error)
            
            # Print epoch statistics
            print(f"\nEpoch {epoch+1} Statistics:")
            print(f"  Average Return: {avg_return:.6f}")
            print(f"  Average Best Cost: {avg_best_cost:.6f}")
            print(f"  Actor Gradient Avg: {avg_actor_grad:.6f}")
            print(f"  Critic Gradient Avg: {avg_critic_grad:.6f}")
            print(f"  Value Prediction Error: {avg_value_error:.6f}")
            
            # Print cumulative action counts (normalized distribution)
            total_destroy_acts = destroy_action_counts.sum().item()
            total_repair_acts = repair_action_counts.sum().item()
            destroy_dist = (destroy_action_counts / total_destroy_acts).cpu().numpy() if total_destroy_acts > 0 else np.zeros(5)
            repair_dist = (repair_action_counts / total_repair_acts).cpu().numpy() if total_repair_acts > 0 else np.zeros(4)
            print(f"  Cumulative Destroy Actions Dist: {destroy_dist.tolist()}")
            print(f"  Cumulative Repair Actions Dist: {repair_dist.tolist()}")
            print(f"  Total Destroy Actions: {int(total_destroy_acts)}")
            print(f"  Total Repair Actions: {int(total_repair_acts)}")
        scheduler.step()
        
        # --- Checkpoint Saving --- 
        if (epoch + 1) % 100 == 0:
            ckpt_path_actor = os.path.join(checkpoint_dir, f"actor_c{num_customers}_v{vehicle_capacity}_epoch_{epoch+1}.pth")
            ckpt_path_critic = os.path.join(checkpoint_dir, f"critic_c{num_customers}_v{vehicle_capacity}_epoch_{epoch+1}.pth")
            torch.save(actor_model.state_dict(), ckpt_path_actor)
            torch.save(critic_model.state_dict(), ckpt_path_critic)
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_dir}")
        # --- End Checkpoint Saving --- 
            
        epoch_total_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Time: {epoch_total_time:.2f}s")
        print(f"  Avg Batch Training Time: {epoch_training_time / epoch_batches:.2f}s")

    # --- Final Model Saving --- 
    final_path_actor = os.path.join(checkpoint_dir, f"actor_c{num_customers}_v{vehicle_capacity}_final.pth")
    final_path_critic = os.path.join(checkpoint_dir, f"critic_c{num_customers}_v{vehicle_capacity}_final.pth")
    torch.save(actor_model.state_dict(), final_path_actor)
    torch.save(critic_model.state_dict(), final_path_critic)
    print(f"Saved final models to {checkpoint_dir}")
    # --- End Final Model Saving --- 
    
    # --- Save Action Counts to YAML --- 
    if collect_stats:
        # Calculate distributions (handle division by zero)
        total_destroy = destroy_action_counts.sum().item()
        total_repair = repair_action_counts.sum().item()
        
        destroy_dist_list = (destroy_action_counts / total_destroy).cpu().tolist() if total_destroy > 0 else [0.0] * 5
        repair_dist_list = (repair_action_counts / total_repair).cpu().tolist() if total_repair > 0 else [0.0] * 4

        # Define operator names (consistent with state_transition.py)
        destroy_op_names = [
            "RandomRemoval", "DemandRelatedRemoval", "GeographicRelatedRemoval", 
            "RouteRelatedRemoval", "GreedyRemoval"
        ]
        repair_op_names = [
            "GreedyRepair", "SortedGreedyRepair", "RegretkRepair(k=2)", "RegretkRepair(k=3)"
        ]

        # Create dictionaries with operator names as keys
        destroy_distribution_dict = {name: dist for name, dist in zip(destroy_op_names, destroy_dist_list)}
        repair_distribution_dict = {name: dist for name, dist in zip(repair_op_names, repair_dist_list)}
        
        action_data = {
            'destroy_distribution': destroy_distribution_dict,
            'repair_distribution': repair_distribution_dict,
            'total_destroy_actions': total_destroy,
            'total_repair_actions': total_repair
        }
        yaml_path = os.path.join(checkpoint_dir, f"action_distribution_c{num_customers}_v{vehicle_capacity}.yaml") # Renamed file
        with open(yaml_path, 'w') as f:
            yaml.dump(action_data, f, default_flow_style=None, sort_keys=False) # Use default flow for dicts
        print(f"Saved action distribution to {yaml_path}")
    # --- End Save Action Counts --- 

    print("\nOverall Training Summary:")
    print(f"  Avg Data Generation Time per Epoch: {total_batch_creation_time / epochs:.2f}s")
    
    # Print training progression - only if collecting stats
    if collect_stats:
        print("\nTraining Progression:")
        for i in range(len(epoch_rewards)):
            print(f"Epoch {i+1}:")
            print(f"  Return: {epoch_rewards[i]:.6f}")
            print(f"  Best Cost: {epoch_best_costs[i]:.6f}")
            print(f"  Actor Gradient: {epoch_actor_grads[i]:.6f}")
            print(f"  Critic Gradient: {epoch_critic_grads[i]:.6f}")
            print(f"  Value Error: {epoch_value_errors[i]:.6f}")

if __name__ == "__main__":
    # Set to False to disable statistics collection and potentially speed up training
    # Set seed for reproducibility
    train(collect_stats=True, seed=42)

