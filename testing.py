import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO  # or your chosen algorithm
from bpo_env import BPOEnv

# Load your trained model
model_path = r'C:\Users\20191955\OneDrive - TU Eindhoven\Desktop\School\Y3Q3\2IOI0\bpo-prefix\tmp\parallel_1000000_25600_sg\best_model.zip'
model = MaskablePPO.load(model_path)

# Define your environment
running_time = 5000  # Set your desired running time
config_type = 'parallel'  # Set your configuration type
allow_postponing = True
reward_function = 'cycle_time'  # Set your reward function
postpone_penalty = 0  # Set your postpone penalty
write_to = 'complete_parallel_10_output'  # Set your output directory

env = BPOEnv(running_time, config_type, allow_postponing, reward_function, postpone_penalty, write_to)
state, _ = env.reset()

# Run the simulation
done = False
total_reward = 0

num_episodes = 50  # Set your desired number of episodes

# Run the simulation for multiple episodes
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Get action from model
        action, _states = model.predict(state, deterministic=True, action_masks=env.action_masks())
        
        # Take action in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Update state
        state = next_state
    
    print(f'Episode {episode + 1}: Total reward: {total_reward}')
print(f'Total reward: {total_reward}')