import csv
import numpy as np
from scipy.optimize import minimize

class FrankaFR3Kinematics:
    def __init__(self):
        # DH Parameters for FR3: [a, alpha, d] (Modified DH convention)
        self.dh_params = [
            (0.0,      0.0,      0.333),
            (0.0,     -np.pi/2,  0.0),
            (0.0,      np.pi/2,  0.316),
            (0.0825,   np.pi/2,  0.0),
            (-0.0825, -np.pi/2,  0.384),
            (0.0,      np.pi/2,  0.0),
            (0.088,    np.pi/2,  0.107)
        ]
        
        # Joint limits for FR3
        self.bounds = [(-2.89, 2.89), (-1.76, 1.76), (-2.89, 2.89), (-3.07, -0.06),
                       (-2.89, 2.89), (0.00, 3.75), (-2.89, 2.89)]

    def _get_transformation_matrix(self, q, a, alpha, d):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(q), np.sin(q)
        return np.array([
            [ct,       -st,      0,    a],
            [st*ca,    ct*ca,   -sa,  -sa*d],
            [st*sa,    ct*sa,    ca,   ca*d],
            [0,         0,       0,    1]
        ])

    def forward_kinematics(self, joints):
        T = np.eye(4)
        for i in range(len(joints)):
            a, alpha, d = self.dh_params[i]
            T = T @ self._get_transformation_matrix(joints[i], a, alpha, d)
        return T

    def inverse_kinematics(self, target_pos, q_guess):
        """Finds a joint configuration for a specific 3D position."""
        def objective(q):
            current_pose = self.forward_kinematics(q)
            # We only optimize for POSITION (x, y, z) here as requested
            pos_err = np.linalg.norm(current_pose[:3, 3] - target_pos)
            # Add a small penalty for moving far from the guess to keep solutions 'natural'
            dist_err = np.linalg.norm(q - q_guess)
            return pos_err + 0.01 * dist_err

        result = minimize(objective, q_guess, method='SLSQP', bounds=self.bounds, tol=1e-6)
        return result.x if result.success else None

    def is_in_collision(self, q, obstacle_bounds):
        """
        Checks if the end-effector is inside a rectangular obstacle.
        obstacle_bounds: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        """
        ee_pos = self.forward_kinematics(q)[:3, 3]
        x_min, x_max = obstacle_bounds['x']
        y_min, y_max = obstacle_bounds['y']
        z_min, z_max = obstacle_bounds['z']
        
        return (x_min <= ee_pos[0] <= x_max and 
                y_min <= ee_pos[1] <= y_max and 
                z_min <= ee_pos[2] <= z_max)

    def get_obstacle_joint_space(self, obstacle_bounds, res_x=3, res_y=3, res_z=3, q_seed=None):
        """
        Samples the rectangular volume and returns list of joint configurations 
        that would put the end-effector inside the obstacle.
        """
        if q_seed is None:
            q_seed = np.zeros(7)
            
        x_range = np.linspace(*obstacle_bounds['x'], res_x)
        y_range = np.linspace(*obstacle_bounds['y'], res_y)
        z_range = np.linspace(*obstacle_bounds['z'], res_z)
        
        forbidden_configs = []
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    target_pos = np.array([x, y, z])
                    q_sol = self.inverse_kinematics(target_pos, q_seed)
                    if q_sol is not None:
                        forbidden_configs.append(q_sol)
                        # Use the last solution as the next seed for smoother mapping
                        q_seed = q_sol
                        
        return forbidden_configs

# --- Usage Example ---
fr3 = FrankaFR3Kinematics()

# Define a rectangular obstacle (Task Space)
# Example: A box in front of the robot
# obs = {
#     'x': ( 0.3,  0.7), 
#     'y': (-0.3, -0.1), 
#     'z': ( 0.0,  0.3)
# }
obs = {
    'x': ( 0.3,  0.7), 
    'y': ( 0.1,  0.3), 
    'z': ( 0.0,  0.3)
}

# # 1. Check if specific joint points are in collision
# points = {
#     "H" : np.array([ 0.000000, -0.785411,  0.000000, -2.356229,  0.000000,  1.570824,  0.785411]),
#     "T1": np.array([ 0.000000,  0.610865,  0.000000, -2.007130,  0.000000,  2.617990,  0.785411]),
#     "T2": np.array([-0.785411,  0.610865,  0.000000, -2.007130,  0.000000,  2.617990,  0.000000]),
#     "T3": np.array([ 0.785411,  0.610865,  0.000000, -2.007130,  0.000000,  2.617990,  1.570800])
# }

# print("Collision Check:")
# for name, q in points.items():
#     collision = fr3.is_in_collision(q, obs)
#     print(f"Point {name}: {'COLLISION' if collision else 'SAFE'}")

# # 2. Get all joint space points corresponding to the obstacle volume
# # resolution=3 means 3x3x3 = 27 samples within the box
# forbidden_q_list = fr3.get_obstacle_joint_space(obs, res_x=5, res_y=5, res_z=30, q_seed=points["H"])

# print(f"\nFound {len(forbidden_q_list)} forbidden joint configurations.")
# if forbidden_q_list:
#     print("Example Forbidden Joint Config (First Sample):")
#     print(np.round(forbidden_q_list[0], 4))

# # --- CSV SAVING SNIPPET ---
# csv_filename = 'O2.csv'
# header = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']

# try:
#     with open(csv_filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(header)      # Write column names
#         writer.writerows(forbidden_q_list) # Write all joint configurations
#     print(f"Successfully saved {len(forbidden_q_list)} configurations to {csv_filename}")
# except Exception as e:
#     print(f"Failed to save CSV: {e}")

import csv
import matplotlib.pyplot as plt

# -------- Read joint angles from CSV --------
csv_file = "obstacles.csv"

joint_trajectories = []
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = np.array([
            float(row['q1']),
            float(row['q2']),
            float(row['q3']),
            float(row['q4']),
            float(row['q5']),
            float(row['q6']),
            float(row['q7'])
        ])
        joint_trajectories.append(q)

joint_trajectories = np.array(joint_trajectories)

# -------- Compute EE positions --------
ee_positions = np.array([
    fr3.forward_kinematics(q)[:3, 3]
    for q in joint_trajectories
])

# -------- Plot EE positions as points only --------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    ee_positions[:, 0],
    ee_positions[:, 1],
    ee_positions[:, 2],
    s=10
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Raw End-Effector Positions (from obstacle.csv)")
ax.grid(True)

plt.show()
