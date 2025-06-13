import gym
from tensorboardX import SummaryWriter
import Config
from Agent import Agent
from TestProcess import TestProcess
from lunar_lander_env import UnrealLunarLanderEnv
from wrappers import NormalizeObservation
import os

# Get the directory where your Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to the executable
YOUR_UNREAL_EXE_PATH = os.path.join(script_dir, "Unreal Engine", "lunar_lander.exe")

# --------------------------------------------------- Initialization ---------------------------------------------------
# Create Unreal Engine environment and add wrappers
env = UnrealLunarLanderEnv(
    flask_port=5000,
    unreal_exe_path=YOUR_UNREAL_EXE_PATH,
    launch_unreal=True, # Set to False if you start UE manually
    ue_launch_args=["-port=5000"]#,"-nullrhi"]
)
env = NormalizeObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

print("Attempting to reset the environment for the first time...")
state, _ = env.reset()
print("Initial reset successful. Starting training.")

# Create agent which will use DDPG to train NNs
agent = Agent(state.shape[0], env.action_space.shape[0])
# Initialize test process which will be occasionally called to test whether goal is met
test_process = TestProcess(state.shape[0], env.action_space.shape[0])
# Create writer for Tensorboard
writer = SummaryWriter(log_dir='content/runs/'+Config.writer_name) if Config.writer_flag else None
print(Config.writer_name)
# ------------------------------------------------------ Training ------------------------------------------------------
for n_step in range(Config.number_of_steps):
    # Check whether we should test the model
    if agent.check_test(test_process, n_step, writer, env):
        break
    
    # Get an action from the agent
    actions = agent.get_action(state, n_step, env)
    
    # Perform a step in the environment
    new_state, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    
    # Store experience in the replay buffer
    agent.add_to_buffer(state, actions, new_state, reward, done)
    
    # Update the agent's networks
    agent.update(n_step)
    
    # Move to the next state
    state = new_state
    
    # If the episode is over, record results and reset
    if done:
        agent.record_results(n_step, writer, env)
        state, _ = env.reset()
    
if writer is not None:
    writer.close()
test_process.env.close()
env.close()