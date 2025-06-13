import Config
from NNs import PolicyNN
import torch
import gym
import numpy as np
from lunar_lander_env import UnrealLunarLanderEnv
from wrappers import NormalizeObservation
import os

class TestProcess:
    def __init__(self, input_state, output_action):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.env = None

    def test(self, writer, trained_policy, env_episode):
        self.policy_nn.load_state_dict(trained_policy.state_dict())
        
        # --- FIX: Create an instance of the custom Unreal Engine environment ---
        # --- IMPORTANT: Use a different port (e.g., 5001) to avoid conflicts with the main training environment ---
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build path to the executable
        YOUR_UNREAL_EXE_PATH = os.path.join(script_dir, "Unreal Engine", "lunar_lander.exe")
        
        self.env = UnrealLunarLanderEnv(
            flask_port=5001,
            unreal_exe_path=YOUR_UNREAL_EXE_PATH,
            launch_unreal=True,
            ue_launch_args=["-port=5000","-nullrhi"]
        )
        # --- FIX: Correctly apply wrappers to self.env ---
        self.env = NormalizeObservation(self.env)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        
        print("\n--- Running Test ---")
        state, _ = self.env.reset()
        
        for n_episode in range(Config.test_episodes):
            while True:
                # During testing, we don't add noise to the actions
                actions = self.policy_nn(torch.Tensor(state).to(self.device))
                
                # --- FIX: Use the new Gym API for step, unpacking 5 values ---
                new_state, reward, terminated, truncated, _ = self.env.step(actions.cpu().detach().numpy())
                done = terminated or truncated
                
                state = new_state
                if done:
                    state, _ = self.env.reset()
                    print('.', end="", flush=True) # Print a dot for each completed test episode
                    break
        
        print("\n--- Test Complete ---")
        mean_return = np.mean(self.env.return_queue)
        
        if writer is not None:
            writer.add_scalar('testing_100_reward', mean_return, env_episode)

        goal_reached = self.check_goal(mean_return)
        
        # Close the test environment window and process
        self.env.close() 
        
        return goal_reached

    def check_goal(self, mean_return):
        if mean_return < 200:
            print("Goal NOT reached! Mean 100 test reward: " + str(np.round(mean_return, 2)))
            return False
        else:
            print("GOAL REACHED! Mean reward over 100 episodes is " + str(np.round(mean_return, 2)))
            torch.save(self.policy_nn.state_dict(), 'models/model' + Config.date_time + '.p')
            return True

    def record_final_episode(self):
        # This function is not called in the main loop, but it has been updated for completeness
        self.env = gym.wrappers.RecordVideo(self.env, "bestRecordings", name_prefix="rl-video" + Config.date_time)
        state, _ = self.env.reset()
        while True:
            actions = self.policy_nn(torch.Tensor(state).to(self.device))
            new_state, reward, terminated, truncated, _ = self.env.step(actions.cpu().detach().numpy())
            state = new_state
            if terminated or truncated:
                break