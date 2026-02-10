#!/usr/bin/env python3
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
            gamma_t = self.gammas(t)
            self.solver.add(gamma_t[2] < 0.1)
            for key, value in enumerate(self.an_exp(t)):
                a_min = self.axis_range_values[f'a{key + 1}_min']
                a_max = self.axis_range_values[f'a{key + 1}_max']
                self.solver.add(z3.And(self.an_exp(t)[value] >= a_min, self.an_exp(t)[value] <= a_max))

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
        # all_points = []
        # dim = self.dimension
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # if dim == 2:
        #     plt.gca().set_aspect('equal')
        #     plt.title("2D Ellipsoid Boundary")
        # elif dim == 3:
        #     ax.set_title("3D Ellipsoid Boundary")
        # else:
        #     ax.set_title(f"{dim}D Ellipsoid (First 3D Projection)")

        # t_values = np.arange(self._start, self._finish, self._step)
        # for t in t_values:
        #     axis_lengths = [self.an_exp_real(t)[f'a{dim+1}'] for dim in range(self.dimension)]
        #     center = self.real_gammas(t, self.C_solved)
        #     points = self.generate_ellipsoid_points_real(axis_lengths, center, step_deg=15)
        #     all_points.append(points)

        # for point_dict in all_points:
        #     self.visualize_real(point_dict, ax)

        # for i in self.obstacles:
        #     ax.add_collection3d(Poly3DCollection(self.faces(i), facecolors='red', edgecolors='r', alpha=0.25))

        # for i in self.setpoints:
        #     ax.add_collection3d(Poly3DCollection(self.faces(i), facecolors='green', edgecolors='green', alpha=0.25))

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
                # self.plot_for_nD(self.C_solved)
                # self.print_equation(self.C_solved)
                end = time.time()
                self.displayTime(start, end)
                # plt.show(block=True)

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
        header = ["Tube Coefficients"] + ["Value"] + list(an_dict.keys())
        num_rows = max(len(C_list), max(len(v) for v in an_dict.values()))

        first_column = []
        first_column.append([f'C_x{dim+1},{j}' for dim in range(self.dimension) for j in range(self.degree + 1)])

        rows = []
        for i in range(num_rows):
            row = []
            row.append(first_column[0][i] if i < len(C_list) else "")
            row.append(C_list[i] if i < len(C_list) else "")
            for key in an_dict:
                vals = an_dict[key]
                row.append(vals[i] if i < len(vals) else "")
            rows.append(row)

        print(tabulate(rows, headers=header, tablefmt='grid'))

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

        for i in range(self.dimension):
            setattr(self, f'set_dim{i}_start', min(coord_lowers[f'dim{i}_min']))
            setattr(self, f'set_dim{i}_finish', max(coord_uppers[f'dim{i}_max']))

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
            x = A @ u + center
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
    start = time.time()
    dim = solver.dimension
    assert len(args) == 2 * dim + 2, f"Expected {2*dim+2} arguments, got {len(args)}"
    bounds_flat = args[:-2]
    t1, t2 = args[-2], args[-1]
    solver.setpoints.append(list(args))

    bounds = [(bounds_flat[i], bounds_flat[i + 1]) for i in range(0, 2 * dim, 2)]

    t_values = np.arange(t1, t2, solver._step)
    theta_grid = solver.sample_theta_grid(step_deg=90)
    all_constraints = []

    for t in t_values:
        gamma_t = solver.gammas(t)

        gamma_constraints = []
        for d in range(dim):
            lower, upper = bounds[d]
            gamma_constraints.append(z3.And(gamma_t[d] > lower, gamma_t[d] < upper))

        all_constraints.append(gamma_constraints)

        for thetas in theta_grid:
            u = solver.generate_u_thetas(thetas)
            boundary_point_constraints = []
            
            for d in range(dim):
                lower, upper = bounds[d]
                P_d = solver.an_exp(t)[f'a{d+1}'] * u[d] + gamma_t[d]

                boundary_point_constraints.append(z3.And(P_d > lower, P_d < upper))

        all_constraints.append(boundary_point_constraints)

    print("Added Reach Constraints: ", solver.setpoints)
    end = time.time()
    solver.displayTime(start, end)
    return all_constraints


# def reach(solver, *args):
    dim = solver.dimension
    bounds_flat = args[:-2]
    t1, t2 = args[-2], args[-1]
    solver.setpoints.append(list(args))

    bounds = [(bounds_flat[i], bounds_flat[i + 1]) for i in range(0, 2 * dim, 2)]
    t_values = np.arange(t1, t2, solver._step)
    all_constraints = []

    for t in t_values:
        gamma_t = solver.gammas(t)
        # Get the symbolic radii for each dimension at time t
        radii = solver.an_exp(t) 

        for d in range(dim):
            lower, upper = bounds[d]
            radius_d = radii[f'a{d+1}']
            center_d = gamma_t[d]

            # Algebraic containment: Center +/- Radius must be within bounds
            all_constraints.append(center_d - radius_d >= lower)
            all_constraints.append(center_d + radius_d <= upper)

    print(f"Added Algebraic Reach Constraints for {len(t_values)} time steps")
    return all_constraints

def avoid(solver, *args):
    """
    args = [x1, x2, y1, y2, ..., t1, t2]
    Total args = 2 * dimension + 2.
    Constraint: The ellipsoid must NOT overlap with the box [x1, x2] x [y1, y2] x ...
    (Note: The standard way is that NO part of the path is within the obstacle set.)
    """
    start = time.time()
    dim = solver.dimension
    assert len(args) == 2 * dim + 2, f"Expected {2*dim+2} arguments, got {len(args)}"
    bounds_flat = args[:-2]
    t1, t2 = args[-2], args[-1]
    solver.obstacles.append(list(args))

    bounds = [(bounds_flat[i], bounds_flat[i + 1]) for i in range(0, 2 * dim, 2)]

    t_values = np.arange(t1, t2, solver._step)
    theta_grid = solver.sample_theta_grid(step_deg=180)
    all_constraints = []

    for t in t_values:
        gamma_t = solver.gammas(t)

        gamma_constraints = []
        for d in range(dim):
            lower, upper = bounds[d]
            gamma_constraints.append(z3.Or(gamma_t[d] < lower, gamma_t[d] > upper))

        all_constraints.append(gamma_constraints)

        for thetas in theta_grid:
            u = solver.generate_u_thetas(thetas)
            boundary_point_constraints = []

            for d in range(dim):
                lower, upper = bounds[d]
                P_d = solver.an_exp(t)[f'a{d+1}'] * u[d] + gamma_t[d]

                boundary_point_constraints.append(z3.Or(P_d < lower, P_d > upper))

        all_constraints.append(boundary_point_constraints)

    for constraints in all_constraints:
        solver.solver.add(constraints)
    # print("Added Avoid Constraints: ", solver.obstacles)
    # end = time.time()
    # solver.displayTime(start, end)
    # return all_constraints

# def real_gammas(t, C_fin):
#         '''method to calculate tube equations'''
#         real_tubes = np.zeros(3)
#         degree = int((len(C_fin) / (3)) - 1)

#         for i in range(3):
#             power = 0
#             for j in range(degree + 1):
#                 real_tubes[i] += ((C_fin[j + i * (degree + 1)]) * (t ** power))
#                 power += 1
#         return real_tubes

# def real_gamma_dot(t, C_fin):
#     '''method to calculate tube equations'''
#     real_tubes = np.zeros(3)
#     degree = int((len(C_fin) / (3)) - 1)

#     for i in range(3):
#         power = 0
#         for j in range(degree + 1):
#             if power < 1:
#                 real_tubes[i] += 0
#                 power += 1
#             else:
#                 real_tubes[i] += power * ((C_fin[j + i * (degree + 1)]) * (t ** (power - 1)))
#                 power += 1
#     return real_tubes

# def tube_plotter(C_array):
#     any_empty = any(t[0] is None or (hasattr(t[0], '__len__') and len(t[0]) == 0) for t in C_array)

#     if not any_empty:
#         fig, axs = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
#         ax, bx, cx= axs

#         for tube in C_array:
#             step = 0.1
#             start = tube[1]
#             end = tube[2]
#             time_range = int((end - start + step)/step)

#             x = np.zeros(time_range)
#             y = np.zeros(time_range)
#             z = np.zeros(time_range)

#             gd_x = np.zeros(time_range)
#             gd_y = np.zeros(time_range)
#             gd_z = np.zeros(time_range)

#             for i in range(time_range):
#                 tube_gamma = real_gammas(start + i * step, tube[0])
#                 x[i] = tube_gamma[0]
#                 y[i] = tube_gamma[1]
#                 z[i] = tube_gamma[2]

#                 tube_gamma_dot = real_gamma_dot(start + i * step, tube[0])
#                 gd_x[i] = tube_gamma_dot[0]
#                 gd_y[i] = tube_gamma_dot[1]
#                 gd_z[i] = tube_gamma_dot[2]

#             t = np.linspace(start, end, time_range)
#             print("range: ", time_range, "\nstart: ", start, "\nfinish: ", end, "\nstep: ", step)

#             ax.plot(t, x)
#             bx.plot(t, y)
#             cx.plot(t, z)

#         plt.show()

def real_gammas(t, C_fin, dim, degree):
    '''Generalized method to calculate tube equations for n dimensions'''
    real_tubes = np.zeros(dim)
    for i in range(dim):
        power = 0
        for j in range(degree + 1):
            # Indexing: i*(degree+1) gets us to the start of the coefficients for dimension i
            real_tubes[i] += ((C_fin[j + i * (degree + 1)]) * (t ** power))
            power += 1
    return real_tubes

def real_gamma_dot(t, C_fin, dim, degree):
    '''Generalized method to calculate tube velocities for n dimensions'''
    real_tubes = np.zeros(dim)
    for i in range(dim):
        power = 0
        for j in range(degree + 1):
            if power < 1:
                real_tubes[i] += 0
            else:
                real_tubes[i] += power * ((C_fin[j + i * (degree + 1)]) * (t ** (power - 1)))
            power += 1
    return real_tubes

def tube_plotter(C_array, dim=7):
    """
    Generalized plotter that automatically detects the polynomial degree 
    for each individual tube in the array.
    """
    # Check if array has data
    if not C_array or C_array[0][0] is None:
        print("No data to plot.")
        return

    # Create subplots for n dimensions
    fig, axs = plt.subplots(dim, 1, figsize=(10, 2 * dim), constrained_layout=True)
    
    # Ensure axs is iterable even if there is only 1 dimension
    if dim == 1:
        axs = [axs]

    for tube_idx, tube in enumerate(C_array):
        coeffs = tube[0]
        start = tube[1]
        end = tube[2]

        # --- NEW: Calculate degree for THIS specific tube ---
        # coeffs length = dim * (degree + 1)
        num_coeffs = len(coeffs)
        if num_coeffs % dim != 0:
            print(f"Error: Tube {tube_idx} coefficients ({num_coeffs}) "
                  f"not divisible by dimension ({dim})")
            continue

        current_degree = int((num_coeffs / dim) - 1)
        print(f"Tube {tube_idx}: Detected polynomial degree = {current_degree}")
        # ----------------------------------------------------

        step = 0.05 
        t_values = np.arange(start, end + step, step)

        # Initialize storage: coords[dimension][time_step]
        coords = [[] for _ in range(dim)]

        for t in t_values:
            # Pass the detected degree to the calculator
            current_pos = real_gammas(t, coeffs, dim, current_degree)
            for d in range(dim):
                coords[d].append(current_pos[d])

        # Plot each dimension
        for d in range(dim):
            label = f"Tube {tube_idx} (Deg {current_degree})"
            axs[d].plot(t_values, coords[d], label=label)
            axs[d].set_ylabel(f"Joint {d+1} (rad)")
            axs[d].grid(True, linestyle='--', alpha=0.6)
            if d == 0:
                axs[d].set_title(f"7-DOF Manipulator Joint Trajectories")
            if d == dim - 1:
                axs[d].set_xlabel("Time (s)")
            axs[d].legend(loc='upper right', fontsize='small')

    plt.show()

def load_forbidden_configs(filename='forbidden_joint_configs.csv'):
    forbidden_list = []
    
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            
            # Skip the header row (q1, q2, ..., q7)
            next(reader)
            
            # Iterate through rows and convert strings to floats
            for row in reader:
                # Convert the list of strings to a list of floats
                joint_config = [float(val) for val in row]
                forbidden_list.append(joint_config)
                
        print(f"Successfully loaded {len(forbidden_list)} configurations.")
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return forbidden_list

_tube_loaded = False
_cached_tubes = []
_tube_filename = "/home/focaslab/ros2_ws/src/STT-for-manipulators/coefficients.csv"

HARD_CODED_TIMESTAMPS = [
    (0, 7),
    (7, 13),
    (13, 24)
]


def load_tube_coefficients():
    global _tube_loaded, _cached_tubes

    if not _tube_loaded:
        try:
            with open(_tube_filename, "r") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    if "Tube Coefficients" in line:
                        _cached_tubes.append([])
                        continue

                    if not _cached_tubes:
                        continue

                    delimiter_pos = line.find('",')
                    if delimiter_pos != -1:
                        value_start = delimiter_pos + 2
                        next_comma = line.find(",", value_start)

                        if next_comma != -1:
                            value_str = line[value_start:next_comma]
                        else:
                            value_str = line[value_start:]

                        try:
                            if value_str:
                                _cached_tubes[-1].append(float(value_str))
                        except ValueError:
                            pass

            _tube_loaded = True

        except OSError:
            print(f"Error: Could not open tube file: [{_tube_filename}]")
            _tube_loaded = True
            return []

    # ---- Apply hard-coded timestamps ----
    tubes_with_time = []

    for i, tube in enumerate(_cached_tubes):
        if i < len(HARD_CODED_TIMESTAMPS):
            start_time, end_time = HARD_CODED_TIMESTAMPS[i]
        else:
            # Fallback if there are more tubes than timestamps
            start_time, end_time = None, None

        tubes_with_time.append([tube, start_time, end_time])

    return tubes_with_time



# --- Usage ---
O_constraints_list = load_forbidden_configs('obstacles.csv')

start = time.time()

#----------------------------------------------------------------------------#
#---------------------------------- TUBE 1 ----------------------------------#
solver1 = STT_Solver(3, 7, 0.5, [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002])

S_constraints_list = reach(solver1, 0.000000,  0.020000,    #    0
                                   -0.785411, -0.765411,    # - 45
                                    0.000000,  0.020000,    #    0
                                   -2.356229, -2.336229,    # -135
                                    0.000000,  0.020000,    #    0
                                    1.570824,  1.590824,    #   90
                                    0.785411,  0.805411,    #   45
                                    0, 1)
T1_constraints_list = reach(solver1, 0.000000, 0.020000,    #    0
                                     0.610865, 0.630865,    #   35
                                     0.000000, 0.020000,    #    0
                                    -2.007130,-1.987130,    # -115
                                     0.000000, 0.020000,    #    0
                                     2.617990, 2.637990,    #  150
                                     0.785411, 0.805411,    #   45
                                     6, 7)

for S in S_constraints_list:
    solver1.solver.add(S)

for T1 in T1_constraints_list:
    solver1.solver.add(T1)

start0 = time.time()
for O in O_constraints_list:
    avoid(solver1, O[0], O[0] + 0.02, 
                    O[1], O[1] + 0.02, 
                    O[2], O[2] + 0.02, 
                    O[3], O[3] + 0.02, 
                    O[4], O[4] + 0.02, 
                    O[5], O[5] + 0.02, 
                    O[6], O[6] + 0.02, 
                    0.0, 7.0)
print("Added Avoid Constraints")
print(f"Time taken:{(time.time() - start0)/60} minutes")

tube1 = solver1.find_solution()

# #----------------------------------------------------------------------------#
# #---------------------------------- TUBE 2 ----------------------------------#
solver2 = STT_Solver(3, 7, 0.1, [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002])

T1_constraints_list = reach(solver2, 0.000000, 0.020000,    #    0
                                     0.610865, 0.630865,    #   35
                                     0.000000, 0.020000,    #    0
                                    -2.007130,-1.987130,    # -115
                                     0.000000, 0.020000,    #    0
                                     2.617990, 2.637990,    #  150
                                     0.785411, 0.805411,    #   45
                                     6, 7)
T2_constraints_list = reach(solver2,-0.785411,-0.765411,    # - 45
                                     0.610865, 0.630865,    #   35
                                     0.000000, 0.020000,    #    0
                                    -2.007130,-1.987130,    # -115
                                     0.000000, 0.020000,    #    0
                                     2.617990, 2.637990,    #  150
                                     0.000000, 0.020000,    #    0
                                     12, 13)

# for T1 in T1_constraints_list:
#     solver2.solver.add(T1)

for T2 in T2_constraints_list:
    solver2.solver.add(T2)

start0 = time.time()
for O in O_constraints_list:
    avoid(solver2, O[0], O[0] + 0.02, 
                    O[1], O[1] + 0.02, 
                    O[2], O[2] + 0.02, 
                    O[3], O[3] + 0.02, 
                    O[4], O[4] + 0.02, 
                    O[5], O[5] + 0.02, 
                    O[6], O[6] + 0.02, 
                    6.0, 13.0)
print("Added Avoid Constraints")
print(f"Time taken:{(time.time() - start)/60} minutes")

solver2.join_constraint(tube1, solver1, 7)
tube2 = solver2.find_solution()

# #----------------------------------------------------------------------------#
# #---------------------------------- TUBE 3 ----------------------------------#
solver3 = STT_Solver(3, 7, 0.5, [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002], [0.002, 0.002])

T2_constraints_list = reach(solver3,-0.785411,-0.765411,    # - 45
                                     0.610865, 0.630865,    #   35
                                     0.000000, 0.020000,    #    0
                                    -2.007130,-1.987130,    # -115
                                     0.000000, 0.020000,    #    0
                                     2.617990, 2.637990,    #  150
                                     0.000000, 0.020000,    #    0
                                     12, 13)

T3_constraints_list = reach(solver3, 0.785411, 0.805411,    #   45
                                     0.610865, 0.630865,    #   35
                                     0.000000, 0.020000,    #    0
                                    -2.007130,-1.987130,    # -115
                                     0.000000, 0.020000,    #    0
                                     2.617990, 2.637990,    #  150
                                     1.570800, 1.590800,    #   90
                                     23, 24)

# for T2 in T2_constraints_list:
#     solver3.solver.add(T2)

for T3 in T3_constraints_list:
    solver3.solver.add(T3)

start0 = time.time()
for O in O_constraints_list:
    avoid(solver3, O[0], O[0] + 0.02, 
                    O[1], O[1] + 0.02, 
                    O[2], O[2] + 0.02, 
                    O[3], O[3] + 0.02, 
                    O[4], O[4] + 0.02, 
                    O[5], O[5] + 0.02, 
                    O[6], O[6] + 0.02, 
                    12.0, 24.0)
print("Added Avoid Constraints")
print(f"Time taken:{(time.time() - start0)/60} minutes")

solver3.join_constraint(tube2, solver2, 13)
tube3 = solver3.find_solution()

# #----------------------------------------------------------------------------#
# #----------------------------------------------------------------------------#

tubes = [[tube1, 0, 7],
         [tube2, 7, 13],
         [tube3, 13, 24]
        ]

print(f"Time taken:{(time.time() - start)/60} minutes")

tube_plotter(tubes)


# tube_plotter(load_tube_coefficients())
