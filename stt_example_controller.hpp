#pragma once
#include <cmath>
#include <string>
#include <cassert>
#include <fstream>
#include <exception>
#include <Eigen/Eigen>
#include <rclcpp/rclcpp.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <controller_interface/controller_interface.hpp>

using namespace std;
using namespace Eigen;
using Vector7d = Matrix<double, 7, 1>;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

    class STTExampleController : public controller_interface::ControllerInterface {
        public:
            [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration() const override;
            [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration() const override;
            controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;
            CallbackReturn on_init() override;
            CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
            CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

        private:
            vector<vector<double>> loadTubeCoefficients();
            vector<double> real_gammas(double t, const vector<double>& C_fin, int degree);
            vector<double> get_tube_state(double t, const vector<vector<double>>& C_array, int dim);
            Vector7d compute_torque_command(const Vector7d& joint_positions_desired, const Vector7d& joint_positions_current, const Vector7d& joint_velocities_current);
            string arm_id_;
            string robot_description_;
            const int num_joints = 7;
            Vector7d q_ = {0.000000, -0.785411, 0.000000, -2.356229, 0.000000, 1.570824, 0.785411};
            Vector7d initial_q_;
            Vector7d dq_;
            Vector7d dq_filtered_;
            double elapsed_time_{0.0};
            void updateJointStates();
    };
}
