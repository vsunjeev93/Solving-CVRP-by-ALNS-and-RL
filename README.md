# VRP-ALNS-RL

A Vehicle Routing Problem (VRP) solver using Adaptive Large Neighborhood Search (ALNS) with Reinforcement Learning.

## Overview

This project implements a hybrid approach combining ALNS with Reinforcement Learning to solve VRP instances. The system uses a Graph Neural Network (GNN) based actor-critic architecture to learn and select optimal destroy and repair operators during the ALNS process.

## Key Components

- **ALNS Operators**:
  - Destroy Operators: RandomRemoval, DemandRelatedRemoval, GeographicRelatedRemoval, RouteRelatedRemoval, GreedyRemoval
  - Repair Operators: GreedyRepair, SortedGreedyRepair, RegretkRepair

- **Neural Network Architecture**:
  - Actor: GNN-based policy network for operator selection
  - Critic: GNN-based value network for state evaluation
  - Uses GIN (Graph Isomorphism Network) for graph processing

- **Core Modules**:
  - `state_transition.py`: Manages state transitions between destroy and repair phases
  - `actor.py` & `critic.py`: Neural network implementations
  - `destroy_actions.py` & `repair_actions.py`: ALNS operator implementations
  - `vrp_data.py`: VRP instance representation and utilities
  - `graph_data.py`: Graph generation, data handling and initial solution generation (nearest neighbor)
  - `train_reinforce.py`: Training loop implementation


## References

- Johnn, S. N., Darvariu, V. A., Handl, J., & Kalcsics, J. (2024). A graph reinforcement learning framework for neural adaptive large neighbourhood search. Computers & Operations Research, 172, 106791.
- Reijnen, Robbert, et al. "Online control of adaptive large neighborhood search using deep reinforcement learning." Proceedings of the International Conference on Automated Planning and Scheduling. Vol. 34. 2024.
