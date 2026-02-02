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

    def _get_transformation_matrix(self, q, a, alpha, d):
        """Modified DH Transformation Matrix."""
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(q), np.sin(q)
        return np.array([
            [ct,       -st,      0,    a],
            [st*ca,    ct*ca,   -sa,  -sa*d],
            [st*sa,    ct*sa,    ca,   ca*d],
            [0,         0,       0,    1]
        ])

    def forward_kinematics(self, joints):
        """Translates Joint Space -> Task Space (4x4 Transform)."""
        T = np.eye(4)
        for i in range(len(joints)):
            a, alpha, d = self.dh_params[i]
            T = T @ self._get_transformation_matrix(joints[i], a, alpha, d)
        return T

    def inverse_kinematics(self, target_pose, q_guess):
        """
        Translates Task Space -> Joint Space.
        Uses Scipy's 'SLSQP' which is compatible with all modern versions.
        """
        def objective(q):
            current_pose = self.forward_kinematics(q)
            # Distance error (Position)
            pos_err = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])
            # Orientation error (simplified trace-based rotation error)
            rot_err = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3])
            return pos_err + 0.5 * rot_err

        # Joint limits for FR3 (approximate)
        bounds = [(-2.89, 2.89), (-1.76, 1.76), (-2.89, 2.89), (-3.07, -0.06),
                  (-2.89, 2.89), (0.00, 3.75), (-2.89, 2.89)]

        result = minimize(objective, q_guess, method='SLSQP', bounds=bounds, tol=1e-6)
        return result.x if result.success else None

# # --- Usage Example ---
# fr3 = FrankaFR3Kinematics()

# # Initial joint configuration
# current_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785])

# # 1. Get current EE Pose
# current_pose = fr3.forward_kinematics(current_q)
# print("Current EE Position:\n", current_pose[:3, 3])

# # 2. Define a goal (move 5cm in X and 5cm in Z)
# goal_pose = current_pose.copy()
# goal_pose[0, 3] += 0.05 
# goal_pose[2, 3] += 0.05

# # 3. Solve IK
# new_q = fr3.inverse_kinematics(goal_pose, current_q)

# if new_q is not None:
#     print("\nNew Joint Angles:\n", new_q)
#     # Verification
#     check_pose = fr3.forward_kinematics(new_q)
#     print("\nVerification (Resulting Position):\n", check_pose[:3, 3])

fr3 = FrankaFR3Kinematics()
H_q = np.array([ 0.000000, -0.785411,  0.000000, -2.356229,  0.000000,  1.570824,  0.785411])
T1_q = np.array([ 0.000000,  0.610865,  0.000000, -2.007130,  0.000000,  2.617990,  0.785411])
T2_q = np.array([-0.785411,  0.610865,  0.000000, -2.007130,  0.000000,  2.617990,  0.000000])
T3_q = np.array([ 0.785411,  0.610865,  0.000000, -2.007130,  0.000000,  2.617990,  1.570800])

H_pose = fr3.forward_kinematics(H_q)
T1_pose = fr3.forward_kinematics(T1_q)
T2_pose = fr3.forward_kinematics(T2_q)
T3_pose = fr3.forward_kinematics(T3_q)

print(f"EE Positions:\nH: {H_pose[:3, 3]} \nT1: {T1_pose[:3, 3]} \nT2: {T2_pose[:3, 3]} \nT3: {T3_pose[:3, 3]}")
