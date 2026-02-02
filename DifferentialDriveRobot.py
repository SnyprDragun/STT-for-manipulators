#!/Users/subhodeep/venv/bin/python3
'''
script for multidimensional hyper ellipsoid stt
\n
format of C:                                            format of An:
+----------+----------------+-----------+               +-----------+----------------+-----------+
|   Coeff  |    dimension   |   degree  |               |   Coeff   |    dimension   |   degree  |
+==========+================+===========+               +===========+================+===========+
|   C_0,0  |        x       |      0    |               |   C_a1,0  |        x       |      0    |
+----------+----------------+-----------+               +-----------+----------------+-----------+
|   C_0,1  |        x       |      1    |               |   C_a1,1  |        x       |      1    |
+----------+----------------+-----------+               +-----------+----------------+-----------+
|   C_1,0  |        y       |      0    |               |   C_a2,0  |        y       |      0    |
+----------+----------------+-----------+               +-----------+----------------+-----------+
|   C_1,1  |        y       |      1    |               |   C_a2,1  |        y       |      1    |
+----------+----------------+-----------+               +-----------+----------------+-----------+
|   C_2,2  |        z       |      2    |               |   C_a3,2  |        z       |      2    |
+----------+----------------+-----------+               +-----------+----------------+-----------+
|     .    |        .       |      .    |               |     .     |        .       |      .    |
|     .    |        .       |      .    |               |     .     |        .       |      .    |
|     .    |        .       |      .    |               |     .     |        .       |      .    |
+----------+----------------+-----------+               +-----------+----------------+-----------+
\n\n
for any An, we consider:
C_an,d = C_an,0 * (t ^ 0) + C_an,1 * (t ^ 1) + C_an,2 * (t ^ 2) + ... + C_an,d * (t ^ d) where d = degree of polynomial
'''
import z3
import csv
import time
import math
import random
import numpy as np
from tabulate import tabulate
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class STT_Solver():
    '''class for generating STT based on constraints on trajectory'''
    def __init__(self, degree, dimension, time_step, *axes_ranges):
        self.setpoints = []
        self.obstacles = []
        self._step = time_step
        self.degree = degree
        self.dimension = dimension

        self._start = 0
        self._finish = 0
        self._range = 0
        self._x_start = 0
        self._x_finish = 0
        self._y_start = 0
        self._y_finish = 0
        self._z_start = 0
        self._z_finish = 0

        self.axis_range_values = {}
        self.axes_ranges = axes_ranges
        for i, axis_range in enumerate(self.axes_ranges):
            self.axis_range_values[f'a{i+1}_min'] = axis_range[0]
            self.axis_range_values[f'a{i+1}_max'] = axis_range[1]

        self.solver = z3.Solver()
        z3.set_param("parallel.enable", True)
        self.C = [z3.Real(f'C_x{axis+1},{i}') for axis in range(self.dimension) for i in range(self.degree + 1)]

        self.An_dict = {}
        for dim in range(self.dimension):
            self.An_dict[f'a{dim + 1}'] = [z3.Real(f'C_a{dim + 1},{i}') for i in range(self.degree + 1)]

        self.C_solved = []
        self.an_solved = {}

        self.flag = False
        self.go_ahead = True

    def gammas(self, t):
        '''method to calculate tube equations'''
        tubes = [z3.Real(f'gamma_{i}') for i in range(self.dimension)]

        for i in range(self.dimension):
            tubes[i] = 0
            power = 0
            for j in range(self.degree + 1):
                tubes[i] += ((self.C[j + i * (self.degree + 1)]) * (t ** power))
                power += 1
        return tubes

    def an_exp(self, t):
        """
        Returns a dictionary mapping each key in An_dict (e.g., 'a1', 'a2', ...) 
        to its symbolic polynomial expression:
        C_key,0 + C_key,1 * t + C_key,2 * t^2 + ... + C_key,d * t^d
        where d = self.degree
        """
        expr_dict = {}
        for key, coeffs in self.An_dict.items():
            expr = sum(coeffs[i] * (t ** i) for i in range(self.degree + 1))
            expr_dict[key] = expr
        return expr_dict

    def an_exp_real(self, t):
        """
        Returns a dictionary mapping each key in An_dict (e.g., 'a1', 'a2', ...) 
        to its symbolic polynomial expression:
        C_key,0 + C_key,1 * t + C_key,2 * t^2 + ... + C_key,d * t^d
        where d = self.degree
        """
        expr_dict = {}
        for key, coeffs in self.an_solved.items():
            expr = sum(coeffs[i] * (t ** i) for i in range(self.degree + 1))
            expr_dict[key] = expr
        return expr_dict

    def real_gammas(self, t, C_fin):
        '''method to calculate tube equations'''
        real_tubes = np.zeros(self.dimension)

        for i in range(self.dimension):
            power = 0
            for j in range(self.degree + 1):
                real_tubes[i] += ((C_fin[j + i * (self.degree + 1)]) * (t ** power))
                power += 1
        return real_tubes

    def gamma_dot(self, t):
        '''method to calculate tube equations'''
        tubes = [z3.Real(f'gammadot_{i}') for i in range(self.dimension)]

        for i in range(self.dimension):
            tubes[i] = 0
            power = 0
            for j in range(self.degree + 1):
                if power < 1:
                    tubes[i] += 0
                    power += 1
                else:
                    tubes[i] += power * ((self.C[j + i * (self.degree + 1)]) * (t ** (power - 1)))
                    power += 1
        return tubes

    def real_gamma_dot(self, t, C_fin):
        '''method to calculate tube equations'''
        real_tubes = np.zeros(self.dimension)

        for i in range(self.dimension):
            power = 0
            for j in range(self.degree + 1):
                if power < 1:
                    real_tubes[i] += 0
                    power += 1
                else:
                    real_tubes[i] += power * ((C_fin[j + i * (self.degree + 1)]) * (t ** (power - 1)))
                    power += 1
        return real_tubes

    def general(self):
        """
        Generates Z3 constraints that ensure each a{i}(t) lies within [a{i}_min, a{i}_max],
        using self.axis_range_values and the expressions in an_expr_list.

        Returns:
            List of Z3 constraints: a{i}_min <= a{i}(t) <= a{i}_max
        """

        for t in np.arange(self._start, self._finish + self._step, self._step):
            for key, value in enumerate(self.an_exp(t)):
                a_min = self.axis_range_values[f'a{key + 1}_min']
                a_max = self.axis_range_values[f'a{key + 1}_max']
                self.solver.add(z3.And(self.an_exp(t)[value] >= a_min, self.an_exp(t)[value] <= a_max))

        """
        Enforces hyper-ellipsoid eccentricity constraints:
            0 < a_i <= a0  (for all i > 0)
            and ordered ratios: (a_i / a0) is non-decreasing
        """
        # a_lengths = [self.An_dict[f'a{i}'][0] for i in range(1, self.dimension)]
        # print("CHECK: ", a_lengths)
        # a0 = a_lengths[0]  # largest axis (major axis)

        # # Ensure all other axes are positive and <= a0
        # for ai in a_lengths[1:]:
        #     self.solver.add(z3.And(ai > 0, ai <= a0))

        # # Ensure (ai / a0) are in non-decreasing order
        # for i in range(2, len(a_lengths)):
        #     num1 = a_lengths[i-1]
        #     num2 = a_lengths[i]
        #     self.solver.add((num1 / a0) <= (num2 / a0))

    def plot_for_nD(self, C_fin):
        n = self.dimension
        T_range = self.getRange()
        t_values = np.linspace(self.getStart(), self.getFinish(), T_range)

        # Initialize coordinate and velocity arrays
        coords = [np.zeros(T_range) for _ in range(n)]
        coord_dots = [np.zeros(T_range) for _ in range(n)]

        for i in range(T_range):
            t = self.getStart() + i * self._step
            gamma = self.real_gammas(t, C_fin)
            gamma_dot = self.real_gamma_dot(t, C_fin)

            for j in range(n):
                coords[j][i] = gamma[j]
                coord_dots[j][i] = gamma_dot[j]

        # Print debug information
        # for j in range(n):
        #     print(f"gamma_dim{j}: ", coords[j])
        #     print(f"gamma_dot_dim{j}: ", coord_dots[j])

        # Create subplots
        fig, axs = plt.subplots(n, 1, figsize=(8, 2 * n), constrained_layout=True)

        # Ensure axs is always iterable
        if n == 1:
            axs = [axs]

        # Plot setpoints and obstacles as rectangles for each dimension
        for j in range(n):
            for sp in self.setpoints:
                t1, t2 = sp[2 * n], sp[2 * n + 1]
                lower = sp[2 * j]
                upper = sp[2 * j + 1]
                rect = patches.Rectangle((t1, lower), t2 - t1, upper - lower,
                                        edgecolor='green', facecolor='none')
                axs[j].add_patch(rect)

            for obs in self.obstacles:
                t1, t2 = obs[2 * n], obs[2 * n + 1]
                lower = obs[2 * j]
                upper = obs[2 * j + 1]
                rect = patches.Rectangle((t1, lower), t2 - t1, upper - lower,
                                        edgecolor='red', facecolor='none')
                axs[j].add_patch(rect)

            axs[j].plot(t_values, coords[j])
            axs[j].set_title(f"t vs x_{j + 1}")
            axs[j].set_xlabel("t")
            axs[j].set_ylabel(f"x_{j + 1}")

        print("range: ", T_range, "\nstart: ", self.getStart(), "\nfinish: ", self.getFinish(), "\nstep: ", self._step)

        #--------------------- 3D Projection ---------------------#
        all_points = []
        dim = self.dimension
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if dim == 2:
            plt.gca().set_aspect('equal')
            plt.title("2D Ellipsoid Boundary")
        elif dim == 3:
            ax.set_title("3D Ellipsoid Boundary")
        else:
            ax.set_title(f"{dim}D Ellipsoid (First 3D Projection)")

        t_values = np.arange(self._start, self._finish, self._step)
        for t in t_values:
            axis_lengths = [self.an_exp_real(t)[f'a{dim+1}'] for dim in range(self.dimension)]
            center = self.real_gammas(t, self.C_solved)
            points = self.generate_ellipsoid_points_real(axis_lengths, center, step_deg=15)
            all_points.append(points)

        for point_dict in all_points:
            self.visualize_real(point_dict, ax)

        for i in self.obstacles:
            ax.add_collection3d(Poly3DCollection(self.faces(i), facecolors='red', edgecolors='r', alpha=0.25))

        for i in self.setpoints:
            ax.add_collection3d(Poly3DCollection(self.faces(i), facecolors='green', edgecolors='green', alpha=0.25))

    def find_solution(self):
        '''method to plot the tubes'''
        start = time.time()
        print("Solving...")

        self.setAll()
        self.general()

        if self.go_ahead == True:
            if self.solver.check() == z3.sat:
                self.flag = True
                model = self.solver.model()

                xi = np.zeros((self.dimension) * (self.degree + 1))
                for i in range(len(self.C)):
                    try:
                        xi[i] = (np.float64(model[self.C[i]].numerator().as_long()))/(np.float64(model[self.C[i]].denominator().as_long()))
                    except AttributeError:
                        xi[i] = np.float64(model[self.C[i]])
                    # print("{} = {}".format(self.C[i], xi[i]))
                    self.C_solved.append(xi[i])

                for key, coeff_list in self.An_dict.items():
                    solved_coeffs = []
                    for coeff in coeff_list:
                        try:
                            float_val = (np.float64(model[coeff].numerator().as_long())) / (np.float64(model[coeff].denominator().as_long()))
                        except AttributeError:
                            float_val = np.float64(model[coeff])
                        solved_coeffs.append(float_val)
                        # print(f"{coeff} = {float_val}")
                    self.an_solved[key] = solved_coeffs

                self.store_csv(self.C_solved, self.an_solved)
                self.plot_for_nD(self.C_solved)
                # self.print_equation(self.C_solved)
                end = time.time()
                self.displayTime(start, end)
                plt.show(block=True)

            else:
                print("No solution found.")
                print("range: ", self.getRange(), "\nstart: ", self.getStart(), "\nfinish: ", self.getFinish(), "\nstep: ", self._step)
                end = time.time()
                self.displayTime(start, end)

        else:
            print("No solution found.")
            print("range: ", self.getRange(), "\nstart: ", self.getStart(), "\nfinish: ", self.getFinish(), "\nstep: ", self._step)
            end = time.time()
            self.displayTime(start, end)

        return self.C_solved

    def store_csv(self, C_list, an_dict):
        """
        Save C_list and an_dict into a CSV file in the specified format.

        Parameters:
        - C_list: list of tube coefficients (e.g., [C_0,0, C_0,1, ..., C_2,3])
        - an_dict: dict of 'a{i}' -> list of coefficients (e.g., {'a1': [...], 'a2': [...], ...})
        - filename: output CSV file name
        """
        filename='coefficients.csv'
        # Prepare header
        header = ["Tube Coefficients"] + ["Value"] + list(an_dict.keys())

        # Determine max number of rows needed
        num_rows = max(len(C_list), max(len(v) for v in an_dict.values()))

        first_column = []
        if self.dimension == 1:
            first_column.append([f'C_x{dim+1},{j}' for dim in range(self.dimension) for j in range(self.degree + 1)])
        elif self.dimension == 2:
            first_column.append([f'C_x{dim+1},{j}' for dim in range(self.dimension) for j in range(self.degree + 1)])
        elif self.dimension == 3:
            first_column.append([f'C_x{dim+1},{j}' for dim in range(self.dimension) for j in range(self.degree + 1)])

        # Prepare rows
        rows = []
        for i in range(num_rows):
            row = []
            # 0th column: C_x/y
            row.append(first_column[0][i] if i < len(C_list) else "")
            # First column: C_list value or empty
            row.append(C_list[i] if i < len(C_list) else "")
            # Remaining columns: values from each a{i}, or empty if out of range
            for key in an_dict:
                vals = an_dict[key]
                row.append(vals[i] if i < len(vals) else "")
            rows.append(row)

        # After building `rows` and `header`
        print(tabulate(rows, headers=header, tablefmt='grid'))

        # Write to CSV
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def print_equation(self, C):
        for i in range(self.dimension):
            print("gamma", i, "= ", end = "")
            power = 0
            for j in range(self.degree + 1):
                print("C", j + i * (self.degree + 1), "* t.^", power, "+ ", end = "")
                power += 1
            print("\n")

    def faces(self, i):
        vertices = [[i[0], i[2], i[4]], [i[1], i[2], i[4]], [i[1], i[3], i[4]], [i[0], i[3], i[4]],  # Bottom face
                    [i[0], i[2], i[5]], [i[1], i[2], i[5]], [i[1], i[3], i[5]], [i[0], i[3], i[5]]]   # Top face

        # Define the 6 faces of the cube using the vertices
        faces = [   [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
                    [vertices[0], vertices[3], vertices[7], vertices[4]]]  # Left face
        return faces

    def setAll(self):
        all_points = self.setpoints + self.obstacles

        # Initialize lists for each dimension's lower and upper bounds
        coord_lowers = {f'dim{i}_min': [] for i in range(self.dimension)}
        coord_uppers = {f'dim{i}_max': [] for i in range(self.dimension)}

        t1, t2 = [], []

        for point in all_points:
            # point format: [x1, x2, y1, y2, z1, z2, ..., t1, t2]
            for i in range(self.dimension):
                coord_lowers[f'dim{i}_min'].append(point[2 * i])
                coord_uppers[f'dim{i}_max'].append(point[2 * i + 1])
            t1.append(point[2 * self.dimension])
            t2.append(point[2 * self.dimension + 1])

        # Set coordinate bounds
        for i in range(self.dimension):
            setattr(self, f'set_dim{i}_start', min(coord_lowers[f'dim{i}_min']))
            setattr(self, f'set_dim{i}_finish', max(coord_uppers[f'dim{i}_max']))

        # Set time bounds
        self.setStart(min(t1))
        self.setFinish(max(t2))
        self.setRange(int((self.getFinish() - self.getStart() + self._step) / self._step))

    def displayTime(self, start, end):
        k = int(end - start)
        days = k // (3600 * 24)
        hrs = (k // 3600) - (days * 24)
        mins = (k // 60) - (hrs * 60)
        if end - start < 1:
            secs = (((end - start) * 10000) // 100) / 100
        else:
            secs = k - (mins * 60) - (hrs * 3600) - (days * 24 * 3600)
        print("Time taken: ", days, "days, ", hrs , "hours, ", mins, "minutes, ", secs, "seconds")

    def getStart(self):
        return self._start

    def setStart(self, start_value):
        self._start = start_value

    def getFinish(self):
        return self._finish

    def setFinish(self, finish_value):
        self._finish = finish_value

    def getRange(self):
        return self._range

    def setRange(self, value):
        self._range = value

    def generate_u_thetas(self, thetas):
        """
        Compute u(theta) in n-D hyperspherical coordinates (all numerical).
        Input: thetas = [θ1, θ2, ..., θ_{n-1}]
        Output: u (unit vector on hypersphere)
        """
        n = len(thetas) + 1
        u = np.zeros(n)

        for i in range(n):
            prod = 1
            for j in range(i):
                prod *= np.sin(thetas[j])
            if i < n - 1:
                prod *= np.cos(thetas[i])
            u[i] = prod
        return u

    def sample_theta_grid(self, step_deg=15):
        """
        Samples angles in hyperspherical space.
        θ₁ to θₙ₋₂ ∈ [0, π], θₙ₋₁ ∈ [0, 2π]
        """
        theta_ranges = [np.deg2rad(np.arange(0, 180 + step_deg, step_deg))] * (self.dimension - 2)
        theta_ranges.append(np.deg2rad(np.arange(0, 360, step_deg)))  # θₙ₋₁ ∈ [0, 2π)

        return product(*theta_ranges)

    def generate_u_thetas_real(self, thetas):
        """
        Compute u(theta) in n-D hyperspherical coordinates.
        Input: thetas = [θ1, θ2, ..., θ_{n-1}]
        Output: u (unit vector on hypersphere)
        """
        n = len(thetas) + 1
        u = np.zeros(n)
        
        for i in range(n):
            prod = 1
            for j in range(i):
                prod *= np.sin(thetas[j])
            if i < n - 1:
                prod *= np.cos(thetas[i])
            u[i] = prod
        return u

    def sample_theta_grid_real(self, step_deg=15):
        """
        Samples angles in hyperspherical space.
        θ₁ to θₙ₋₂ ∈ [0, π], θₙ₋₁ ∈ [0, 2π]
        """
        theta_ranges = [np.deg2rad(np.arange(0, 180 + step_deg, step_deg))] * (self.dimension - 2)
        theta_ranges.append(np.deg2rad(np.arange(0, 360, step_deg)))  # Last theta in [0, 2π)

        return product(*theta_ranges)

    def generate_ellipsoid_points_real(self, axis_lengths, center = None, step_deg=15):
        """
        axis_lengths: list or array of semi-axis lengths [a1, a2, ..., an]
        step_deg: angular sampling resolution in degrees
        """
        n = len(axis_lengths)
        A = np.diag(axis_lengths)

        theta_grid = self.sample_theta_grid_real(step_deg)
        points = []

        if center is None:
            center = np.zeros(n)
        else:
            center = np.array(center)
            assert len(center) == n, "Center must have same dimension as axis_lengths"

        for thetas in theta_grid:
            u = self.generate_u_thetas_real(thetas)
            x = A @ u + center # Ellipsoid point
            points.append(x)

        return np.array(points)

    def visualize_real(self, points, ax):
        """
        Visualize 2D or 3D projection of high-dimensional ellipsoid.
        """
        dim = self.dimension
        if dim == 2:
            plt.plot(points[:,0], points[:,1], 'o', markersize=2)
        elif dim == 3:
            ax.scatter(points[:,0], points[:,1], points[:,2], s=2)
        else:
            # For n > 3, show first 3 dims as 3D projection
            ax.scatter(points[:,0], points[:,1], points[:,2], s=2)

    def join_constraint(self, prev_tube, prev_solver, prev_t_end):
        if prev_solver.flag == True:
            for i in range(self.dimension):
                self.solver.add(self.gammas(prev_t_end)[i] == prev_solver.real_gammas(prev_t_end, prev_tube)[i])
                self.solver.add(self.gamma_dot(prev_t_end)[i] == prev_solver.real_gamma_dot(prev_t_end, prev_tube)[i])
                self.solver.add(self.an_exp(prev_t_end)[f'a{i+1}'] == prev_solver.an_exp_real(prev_t_end)[f'a{i+1}'])
        else:
            self.go_ahead = False
            print("Previous solver has no solution, cannot join constraints.")


def reach(solver, *args):
    """
    args = [x1, x2, y1, y2, ..., t1, t2]
    Total args = 2 * dimension + 2.
    Constraint: The ellipsoid must be fully contained within the box [x1, x2] x [y1, y2] x ...
    """
    start = time.time() # Assuming start is defined here for timing
    dim = solver.dimension
    assert len(args) == 2 * dim + 2, f"Expected {2*dim+2} arguments, got {len(args)}"
    bounds_flat = args[:-2]  # all spatial bounds
    t1, t2 = args[-2], args[-1]
    solver.setpoints.append(list(args))

    # Convert bounds into [(min, max), (min, max), ...] for each dimension
    bounds = [(bounds_flat[i], bounds_flat[i + 1]) for i in range(0, 2 * dim, 2)]

    t_values = np.arange(t1, t2, solver._step)
    theta_grid = solver.sample_theta_grid(step_deg=15)
    all_constraints = []

    for t in t_values:
        gamma_t = solver.gammas(t)
        
        # 1. Constraint for the center point (gamma)
        gamma_constraints = []
        for d in range(dim):
            lower, upper = bounds[d]
            # Must satisfy: lower < gamma_d < upper for ALL dimensions d
            gamma_constraints.append(z3.And(gamma_t[d] > lower, gamma_t[d] < upper))
        
        # Combine all dimensional constraints for the center point into a single z3.And
        all_constraints.append(z3.And(gamma_constraints))

        # 2. Constraint for sampled points on the ellipsoid boundary
        for thetas in theta_grid:
            u = solver.generate_u_thetas(thetas)  # numerical vector for unit sphere
            boundary_point_constraints = []
            
            # The coordinates of a boundary point P are P_d = a_d * u_d + gamma_d
            for d in range(dim):
                lower, upper = bounds[d]
                # P_d expression: a_{d+1} is used because 'a' indexing starts at 1 in solver.an_exp
                P_d = solver.an_exp(t)[f'a{d+1}'] * u[d] + gamma_t[d]
                
                # Must satisfy: lower < P_d < upper for ALL dimensions d for this point P
                boundary_point_constraints.append(z3.And(P_d > lower, P_d < upper))

            # Combine all dimensional constraints for this boundary point into a single z3.And
            all_constraints.append(z3.And(boundary_point_constraints))

    print("Added Reach Constraints: ", solver.setpoints)
    end = time.time()
    solver.displayTime(start, end)
    return all_constraints

def avoid(solver, *args):
    """
    args = [x1, x2, y1, y2, ..., t1, t2]
    Total args = 2 * dimension + 2.
    Constraint: The ellipsoid must NOT overlap with the box [x1, x2] x [y1, y2] x ...
    (Note: The standard way is that NO part of the path is within the obstacle set.)
    """
    start = time.time() # Assuming start is defined here for timing
    dim = solver.dimension
    assert len(args) == 2 * dim + 2, f"Expected {2*dim+2} arguments, got {len(args)}"
    bounds_flat = args[:-2]  # all spatial bounds
    t1, t2 = args[-2], args[-1]
    solver.obstacles.append(list(args))

    # Convert bounds into [(min, max), (min, max), ...] for each dimension
    bounds = [(bounds_flat[i], bounds_flat[i + 1]) for i in range(0, 2 * dim, 2)]

    t_values = np.arange(t1, t2, solver._step)
    theta_grid = solver.sample_theta_grid(step_deg=15)
    all_constraints = []

    for t in t_values:
        gamma_t = solver.gammas(t)

        # 1. Constraint for the center point (gamma)
        gamma_constraints = []
        for d in range(dim):
            lower, upper = bounds[d]
            # Must satisfy: gamma_d < lower OR gamma_d > upper for AT LEAST ONE dimension d
            gamma_constraints.append(z3.Or(gamma_t[d] < lower, gamma_t[d] > upper))

        # Combine dimensional checks with z3.Or: The center is outside the box if it fails 
        # containment in at least one dimension.
        all_constraints.append(z3.Or(gamma_constraints))

        # 2. Constraint for sampled points on the ellipsoid boundary
        for thetas in theta_grid:
            u = solver.generate_u_thetas(thetas)  # numerical vector for unit sphere
            boundary_point_constraints = []

            # The coordinates of a boundary point P are P_d = a_d * u_d + gamma_d
            for d in range(dim):
                lower, upper = bounds[d]
                # P_d expression: a_{d+1} is used because 'a' indexing starts at 1 in solver.an_exp
                P_d = solver.an_exp(t)[f'a{d+1}'] * u[d] + gamma_t[d]
                
                # Must satisfy: P_d < lower OR P_d > upper for AT LEAST ONE dimension d
                boundary_point_constraints.append(z3.Or(P_d < lower, P_d > upper))

            # Combine dimensional checks with z3.Or: The boundary point is outside the box if it fails 
            # containment in at least one dimension.
            all_constraints.append(z3.Or(boundary_point_constraints))

    print("Added Avoid Constraints: ", solver.obstacles)
    end = time.time()
    solver.displayTime(start, end)
    return all_constraints

def real_gammas(t, C_fin):
        '''method to calculate tube equations'''
        real_tubes = np.zeros(3)
        degree = int((len(C_fin) / (3)) - 1)

        for i in range(3):
            power = 0
            for j in range(degree + 1): #each tube eq has {degree+1} terms
                real_tubes[i] += ((C_fin[j + i * (degree + 1)]) * (t ** power))
                power += 1
        return real_tubes

def real_gamma_dot(t, C_fin):
    '''method to calculate tube equations'''
    real_tubes = np.zeros(3)
    degree = int((len(C_fin) / (3)) - 1)

    for i in range(3):
        power = 0
        for j in range(degree + 1):
            if power < 1:
                real_tubes[i] += 0
                power += 1
            else:
                real_tubes[i] += power * ((C_fin[j + i * (degree + 1)]) * (t ** (power - 1)))
                power += 1
    return real_tubes

def tube_plotter(C_array):
    any_empty = any(t[0] is None or (hasattr(t[0], '__len__') and len(t[0]) == 0) for t in C_array)

    if not any_empty:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
        ax, bx, cx= axs

        for tube in C_array:
            step = 0.1
            start = tube[1]
            end = tube[2]
            time_range = int((end - start + step)/step)

            x = np.zeros(time_range)
            y = np.zeros(time_range)
            z = np.zeros(time_range)

            gd_x = np.zeros(time_range)
            gd_y = np.zeros(time_range)
            gd_z = np.zeros(time_range)

            for i in range(time_range):
                tube_gamma = real_gammas(start + i * step, tube[0])
                x[i] = tube_gamma[0]
                y[i] = tube_gamma[1]
                z[i] = tube_gamma[2]

                tube_gamma_dot = real_gamma_dot(start + i * step, tube[0])
                gd_x[i] = tube_gamma_dot[0]
                gd_y[i] = tube_gamma_dot[1]
                gd_z[i] = tube_gamma_dot[2]

            t = np.linspace(start, end, time_range)
            print("range: ", time_range, "\nstart: ", start, "\nfinish: ", end, "\nstep: ", step)

            ax.plot(t, x)
            bx.plot(t, y)
            cx.plot(t, z)

        plt.show()


start = time.time()

#----------------------------------------------------------------------------#
#---------------------------------- TUBE 1 ----------------------------------#
solver1 = STT_Solver(2, 2, 0.1, [0.5, 0.5], [0.5, 0.5])

S_constraints_list = reach(solver1, 0, 3, 0, 3, 0, 1)
T1_constraints_list = reach(solver1, 6, 9, 6, 9, 6, 7)
T2_constraints_list = reach(solver1, 12, 15, 6, 9, 6, 7)
O_constraints_list = avoid(solver1, 9, 12, 6, 9, 0, 15)

for S in S_constraints_list:
    solver1.solver.add(S)

for O in O_constraints_list:
    solver1.solver.add(O)

T_choice = random.randint(1, 2)
if T_choice == 1:
    print("Choosing T1")
    for T1 in T1_constraints_list:
        solver1.solver.add(T1)
else:
    print("Choosing T2")
    for T2 in T2_constraints_list:
        solver1.solver.add(T2)

tube1 = solver1.find_solution()

#----------------------------------------------------------------------------#
#---------------------------------- TUBE 2 ----------------------------------#
solver2 = STT_Solver(2, 2, 0.1, [0.5, 0.5], [0.5, 0.5])

T1_constraints_list = reach(solver2, 6, 9, 6, 9, 6, 7)
T2_constraints_list = reach(solver2, 12, 15, 6, 9, 6, 7)
G_constraints_list = reach(solver2, 18, 21, 15, 18, 14, 15)
O_constraints_list = avoid(solver2, 9, 12, 6, 9, 0, 15)

for O in O_constraints_list:
    solver2.solver.add(O)

for G in G_constraints_list:
    solver2.solver.add(G)

if T_choice == 1:
    print("Choosing T1")
    for T1 in T1_constraints_list:
        solver2.solver.add(T1)
else:
    print("Choosing T2")
    for T2 in T2_constraints_list:
        solver2.solver.add(T2)

solver2.join_constraint(tube1, solver1, 6)
tube2 = solver2.find_solution()

#----------------------------------------------------------------------------#
#---------------------------------- TUBE 3 ----------------------------------#
solver3 = STT_Solver(3, 2, 0.1, [0.5, 0.5], [0.5, 0.5])

G_constraints_list = reach(solver3, 18, 21, 15, 18, 14, 20)

for G in G_constraints_list:
    solver3.solver.add(G)

solver3.join_constraint(tube2, solver2, 14)
tube3 = solver3.find_solution()
#----------------------------------------------------------------------------#

print(time.time() - start, "seconds")

tubes = [[tube1, 0, 7],
         [tube2, 6, 15],
         [tube3, 14, 18]
        ]

def real_gammas(t, C_fin):
        '''method to calculate tube equations'''
        real_tubes = np.zeros(2)
        degree = int((len(C_fin) / (2)) - 1)

        for i in range(2):
            power = 0
            for j in range(degree + 1): #each tube eq has {degree+1} terms
                real_tubes[i] += ((C_fin[j + i * (degree + 1)]) * (t ** power))
                power += 1
        return real_tubes

def real_gamma_dot(t, C_fin):
    '''method to calculate tube equations'''
    real_tubes = np.zeros(2)
    degree = int((len(C_fin) / (2)) - 1)

    for i in range(2):
        power = 0
        for j in range(degree + 1):
            if power < 1:
                real_tubes[i] += 0
                power += 1
            else:
                real_tubes[i] += power * ((C_fin[j + i * (degree + 1)]) * (t ** (power - 1)))
                power += 1
    return real_tubes

def tube_plotter(C_array):
    any_empty = any(t[0] is None or (hasattr(t[0], '__len__') and len(t[0]) == 0) for t in C_array)

    if not any_empty:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
        ax, bx = axs

        for tube in C_array:
            step = 0.1
            start = tube[1]
            end = tube[2]
            time_range = int((end - start + step)/step)

            x = np.zeros(time_range)
            y = np.zeros(time_range)

            gd_x = np.zeros(time_range)
            gd_y = np.zeros(time_range)

            for i in range(time_range):
                tube_gamma = real_gammas(start + i * step, tube[0])
                x[i] = tube_gamma[0]
                y[i] = tube_gamma[1]

                tube_gamma_dot = real_gamma_dot(start + i * step, tube[0])
                gd_x[i] = tube_gamma_dot[0]
                gd_y[i] = tube_gamma_dot[1]

            t = np.linspace(start, end, time_range)
            print("range: ", time_range, "\nstart: ", start, "\nfinish: ", end, "\nstep: ", step)

            ax.plot(t, x)
            bx.plot(t, y)

        plt.show()

tube_plotter(tubes)
