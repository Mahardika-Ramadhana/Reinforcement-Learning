# RL Playground - Snake AI

Learning Reinforcement Learning by building an autonomous agent that plays Snake.

## Project 1: Snake AI (Deep Q-Learning)

Teaching an agent to play Snake using Deep Q-Learning (DQN). The agent learns to navigate the grid, collect food, and avoid collisions with walls and its own body through trial and error.

## Features & Implementation

### 1. State Representation (24 Inputs)
The agent "sees" the environment through a 24-dimensional state vector:
- **Instant Danger (4):** Detection of immediate obstacles in 4 directions (Up, Right, Down, Left).
- **Movement Direction (4):** One-hot encoding of the current direction.
- **Food Location (4):** Relative position of the food (Up, Right, Down, Left).
- **Vision Sensors (8):** Normalized distance sensors scanning in 8 directions (N, S, E, W, NE, NW, SE, SW) to detect walls or body parts.
- **Tail Location (4):** Relative position of the tail to help the agent avoid trapping itself.

### 2. Neural Network Architecture
- **Type:** Linear Deep Q-Network (DQN)
- **Structure:** 
    - Input Layer: 24 nodes
    - Hidden Layer 1: 256 nodes (ReLU)
    - Hidden Layer 2: 256 nodes (ReLU)
    - Output Layer: 3 nodes (Straight, Turn Right, Turn Left)
- **Hardware Acceleration:** Automatic CUDA/GPU support if available.

### 3. Reward System
- **Food (+50):** High positive reward for eating food.
- **Collision (-10):** High negative reward for hitting a wall or itself.
- **Idle (-0.05):** Small negative reward for every step to encourage efficiency.
- **Timeout (-10):** Penalty if the agent spends too much time without eating.

### 4. Training Parameters
- **Algorithm:** Q-Learning with Experience Replay.
- **Optimizer:** Adam (Learning Rate: 0.001).
- **Loss Function:** Mean Squared Error (MSE).
- **Exploration:** Epsilon-Greedy strategy with decay.

## Progress Roadmap

- [x] Pygame & Window setup
- [x] Initial State, Action, and Reward implementation
- [x] Food spawning system
- [x] Snake growth and body logic
- [x] Integrate with Reinforcement Learning algorithms (DQN)
- [x] Implement Vision Sensors (8 directions)
- [x] Add GPU support for training

## How to Run

1. Clone this repository:
   `git clone https://github.com/Mahardika-Ramadhana/Reinforcement-Learning`
2. Create a virtual environment:
   `python3 -m venv venv`
3. Activate the environment:
   `source venv/bin/activate`
4. Install dependencies:
   `pip install torch pygame numpy`
5. Run the training:
   `python3 agent.py`
