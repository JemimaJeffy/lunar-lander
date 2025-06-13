import gym
from gym.spaces import Box
import numpy as np
from flask import Flask, request, jsonify
import threading
import queue
import time
import subprocess
import os
import platform

STATE_SIZE = 17
ACTION_SIZE = 6
MAX_EPISODE_STEPS = 6000

# --- Constants ---
REWARD_SUCCESSFUL_LANDING = 250.0
PENALTY_CRASH = -200.0
PENALTY_TILTED_LANDING = -100.0
PENALTY_OUT_OF_BOUNDS = -150.0
LOW_ALTITUDE_THRESHOLD = 0.1
MAX_LANDING_SPEED_VERTICAL = 0.75
MAX_LANDING_SPEED_HORIZONTAL = 0.75
MAX_ANGULAR_SPEED = np.pi * 2 
ORIENTATION_PITCH_THRESHOLD = np.deg2rad(10)
ORIENTATION_ROLL_THRESHOLD = np.deg2rad(10)
TILT_CRASH_PITCH_THRESHOLD = np.deg2rad(45)
TILT_CRASH_ROLL_THRESHOLD = np.deg2rad(45)
BOUNDS_XY = 100000.0 # In cm, so 100 meters
BOUNDS_Z_MAX = 10000.0


class UnrealLunarLanderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, flask_port=5000, host='0.0.0.0',
                 unreal_exe_path=None, launch_unreal=False, ue_launch_args=None):
        super(UnrealLunarLanderEnv, self).__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        
        MAX_ANG_VEL_BOUND = np.deg2rad(720)
        low_bounds = np.array(
            [-BOUNDS_XY] * 3 + [-np.pi] * 3 + [-1000] * 3 +
            [-MAX_ANG_VEL_BOUND] * 3 + [-10] * 4 + [0.0] * 1, dtype=np.float32
        )
        high_bounds = np.array(
            [BOUNDS_XY] * 3 + [np.pi] * 3 + [1000] * 3 +
            [MAX_ANG_VEL_BOUND] * 3 + [BOUNDS_Z_MAX] * 4 + [1.0] * 1, dtype=np.float32
        )
        self.observation_space = Box(low=low_bounds, high=high_bounds, shape=(STATE_SIZE,), dtype=np.float32)

        self.flask_app = Flask(__name__)
        self.host = host
        self.port = flask_port
        self.server_thread = None
        self.data_from_unreal = queue.Queue(maxsize=1)
        self.action_for_unreal = queue.Queue(maxsize=1)
        self.unreal_exe_path = unreal_exe_path
        self.launch_unreal = launch_unreal
        self.ue_launch_args = ue_launch_args if ue_launch_args is not None else []
        self.unreal_process = None
        self.current_state = np.zeros(STATE_SIZE, dtype=np.float32)
        self.episode_step_count = 0

        self.curriculum_level = 1
        self.average_score_for_curriculum = -np.inf 
        
        self._setup_flask_routes()
        self._start_flask_server()
        if self.launch_unreal: self._launch_unreal_engine()
        print(f"UnrealLunarLanderEnv initialized. Flask server running on http://{self.host}:{self.port}")

    def set_curriculum_level(self, avg_score):
        self.average_score_for_curriculum = avg_score
        new_level = self.curriculum_level
        
        if avg_score > 220 and self.curriculum_level == 3: # Need a high, stable score to advance
            new_level = 4
        elif avg_score > 180 and self.curriculum_level < 3:
            new_level = 3
        elif avg_score > 50 and self.curriculum_level < 2:
            new_level = 2
            
        if new_level != self.curriculum_level:
            self.curriculum_level = new_level
            print(f"\n****** Curriculum level ADVANCED to: {self.curriculum_level} (Avg score: {avg_score:.2f}) ******\n")

    def _launch_unreal_engine(self):
        if not self.unreal_exe_path or not os.path.exists(self.unreal_exe_path): return
        try:
            cmd = [self.unreal_exe_path] + self.ue_launch_args
            self.unreal_process = subprocess.Popen(cmd)
            print(f"UE process started (PID: {self.unreal_process.pid}).")
        except Exception as e:
            self.unreal_process = None
            print(f"[Error] Failed to launch UE: {e}")

    def _setup_flask_routes(self):
        @self.flask_app.route('/control', methods=['POST'])
        def control_lander_route():
            try:
                data = request.json
                self.data_from_unreal.put(data)
                command_from_agent = self.action_for_unreal.get(timeout=10.0)
                return jsonify(command_from_agent)
            except Exception as e:
                return jsonify({"command": "dummy", "error": str(e)}), 500

    def _start_flask_server(self):
        self.server_thread = threading.Thread(target=lambda: self.flask_app.run(host=self.host, port=self.port, debug=False, use_reloader=False))
        self.server_thread.daemon = True
        self.server_thread.start()

    def _calculate_reward_and_done(self, state):
        reward = 0.0
        done = False
        info = {}

        relative_pos_m = state[0:3] / 100.0
        orientation_rad = np.deg2rad(state[3:6])
        lin_vel_ms = state[6:9] / 100.0
        ang_vel_rads = np.deg2rad(state[9:12])
        leg_altitudes_m = state[12:16] / 100.0
        hit_flag = state[16]
        
        lander_altitude_m = np.min(leg_altitudes_m)
        roll, pitch = orientation_rad[0], orientation_rad[1]

        reward -= np.linalg.norm(relative_pos_m) * 0.1 

        if abs(pitch) > ORIENTATION_PITCH_THRESHOLD or abs(roll) > ORIENTATION_ROLL_THRESHOLD:
            reward -= 0.5

        angular_speed_mag = np.linalg.norm(ang_vel_rads)
        if angular_speed_mag > MAX_ANGULAR_SPEED:
            reward -= (angular_speed_mag - MAX_ANGULAR_SPEED) * 0.1

        if self.episode_step_count >= MAX_EPISODE_STEPS:
            reward -= 50.0
            done = True
            info['status'] = 'max_steps_reached'
            return reward, done, info

        if abs(relative_pos_m[0]) > (BOUNDS_XY / 100.0) or abs(relative_pos_m[1]) > (BOUNDS_XY / 100.0):
            reward += PENALTY_OUT_OF_BOUNDS
            done = True
            info['status'] = 'out_of_bounds'
            return reward, done, info

        if hit_flag > 0:
            is_severely_tilted = abs(pitch) > TILT_CRASH_PITCH_THRESHOLD or abs(roll) > TILT_CRASH_ROLL_THRESHOLD
            is_body_hit = lander_altitude_m > LOW_ALTITUDE_THRESHOLD * 1.5

            if is_severely_tilted or is_body_hit:
                reward += PENALTY_CRASH
                done = True
                info['status'] = 'crash_body_or_tilted'
            else:
                horizontal_speed = np.linalg.norm(lin_vel_ms[0:2])
                vertical_speed = abs(lin_vel_ms[2])
                is_too_fast = vertical_speed > MAX_LANDING_SPEED_VERTICAL or horizontal_speed > MAX_LANDING_SPEED_HORIZONTAL
                
                if is_too_fast:
                    reward += PENALTY_CRASH
                    done = True
                    info['status'] = 'crash_hard_landing'
                else:
                    is_on_target = np.linalg.norm(relative_pos_m) < 2.0
                    is_upright = abs(pitch) < ORIENTATION_PITCH_THRESHOLD and abs(roll) < ORIENTATION_ROLL_THRESHOLD
                    if is_on_target and is_upright:
                        reward += REWARD_SUCCESSFUL_LANDING
                        info['status'] = 'successful_landing'
                    else:
                        reward += PENALTY_TILTED_LANDING
                        info['status'] = 'safe_landing_off_target_or_tilted'
                    done = True
        return reward, done, info

    def reset(self):
        while not self.data_from_unreal.empty(): self.data_from_unreal.get_nowait()
        while not self.action_for_unreal.empty(): self.action_for_unreal.get_nowait()
        self.episode_step_count = 0

        if self.launch_unreal and (not self.unreal_process or self.unreal_process.poll() is not None):
            self._launch_unreal_engine()
            if not self.unreal_process:
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
        # --- MODIFIED BLOCK: Generate detailed start_params based on curriculum level ---
        if self.curriculum_level == 1: # Easy: Low, stable, and centered
            start_params = {
                "height": 100.0,
                "horizontal_offset": 5.0,
                "orientation": [0.0, 0.0, 0.0], # roll, pitch, yaw (degrees)
                "linear_velocity": [0.0, 0.0, -1.0], # x, y, z (m/s)
                "angular_velocity": [0.0, 0.0, 0.0] # roll, pitch, yaw rates (deg/s)
            }
        elif self.curriculum_level == 2: # Medium: Higher, some offset and velocity
            start_params = {
                "height": np.random.uniform(150, 300),
                "horizontal_offset": np.random.uniform(10, 40),
                "orientation": [np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-90, 90)],
                "linear_velocity": [np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-3, -1)],
                "angular_velocity": [np.random.uniform(-15, 15), np.random.uniform(-15, 15), np.random.uniform(-15, 15)]
            }
        elif self.curriculum_level == 3:
            start_params = { # This is your previous "hard" level
                "height": np.random.uniform(300, 500),
                "horizontal_offset": np.random.uniform(50, 100),
                "orientation": [np.random.uniform(-45, 45), np.random.uniform(-45, 45), np.random.uniform(-180, 180)],
                "linear_velocity": [np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, -2)],
                "angular_velocity": [np.random.uniform(-45, 45), np.random.uniform(-45, 45), np.random.uniform(-45, 45)]
            }
        else: # Level 4: The Expert Tier
            start_params = {
                "height": np.random.uniform(700, 1500),
                "horizontal_offset": np.random.uniform(100, 300), # Wider horizontal challenge
                "orientation": [np.random.uniform(-60, 60), np.random.uniform(-60, 60), np.random.uniform(-180, 180)],
                "linear_velocity": [np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, -5)], # Higher initial speeds
                "angular_velocity": [np.random.uniform(-60, 60), np.random.uniform(-60, 60), np.random.uniform(-60, 60)]
            }
        # --- END OF MODIFIED BLOCK ---
            
        # IMPORTANT: Your Unreal project must be updated to read these 'params'
        # and set the lander's initial state accordingly.
        reset_command = {"command": "reset", "params": start_params}
        
        try:
            self.action_for_unreal.put(reset_command, timeout=5.0)
        except queue.Full:
            print("[Error] Reset command queue was full. Hard resetting.")
            self.close()
            return self.reset()
            
        print(f"Gym env: RESET (Level {self.curriculum_level}). Waiting for UE to send a 'start' signal...")
        is_ready = False
        initial_state_from_ue = None
        timeout_duration = 120.0
        
        start_time = time.time()
        while not is_ready and (time.time() - start_time) < timeout_duration:
            try:
                ue_data = self.data_from_unreal.get(timeout=1.0)
                if ue_data.get("start") is True and "state" in ue_data:
                    self.action_for_unreal.put({"command": "start_confirmed"})
                    is_ready = True
                    initial_state_from_ue = np.array(ue_data["state"], dtype=np.float32)
                    print("Gym env: 'start' signal received. Starting episode.")
                else:
                    # Keep the line open by sending dummy commands if we get unexpected data
                    self.action_for_unreal.put({"command": "dummy"})
            except queue.Empty:
                continue # Just wait for the next message
            except Exception as e:
                print(f"[Env Reset Error] Handshake error: {e}. Initiating hard reset...")
                self.close()
                return self.reset()

        if not is_ready:
            print(f"Timeout ({timeout_duration}s): No start signal received. Initiating hard reset...")
            self.close()
            return self.reset()
                
        self.current_state = initial_state_from_ue
        return self.current_state, {}

    def step(self, action_payload):
        self.episode_step_count += 1
        action_list = [float(a) for a in action_payload]
        step_command = {"command": "step", "action": action_list}

        try:
            self.action_for_unreal.put(step_command, timeout=2.0)
        except queue.Full:
            reward, _, info = self._calculate_reward_and_done(self.current_state)
            return self.current_state, reward - 50, True, False, {"error": "Action queue full", **info}
            
        try:
            ue_data = self.data_from_unreal.get(timeout=30.0)
            new_state_from_ue = np.array(ue_data["state"], dtype=np.float32)
            
            reward, done, info = self._calculate_reward_and_done(new_state_from_ue)
            self.current_state = new_state_from_ue
            truncated = info.get('status') == 'max_steps_reached'
            return self.current_state, reward, done, truncated, info
        except Exception as e:
            reward, _, info = self._calculate_reward_and_done(self.current_state)
            return self.current_state, reward - 50, True, False, {"error": f"UE response timeout or bad data: {e}", **info}

    def render(self, mode='human'):
        pass

    def close(self):
        if self.unreal_process and self.unreal_process.poll() is None:
            pid = self.unreal_process.pid
            print(f"Terminating UE process tree (PID: {pid})...")
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], check=True, capture_output=True)
                else:
                    self.unreal_process.terminate()
                self.unreal_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, PermissionError):
                self.unreal_process.kill()
            self.unreal_process = None
            print("UE process terminated.")