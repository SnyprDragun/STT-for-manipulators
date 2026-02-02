#include <franka_example_controllers/stt_example_controller.hpp>

namespace franka_example_controllers {

  controller_interface::InterfaceConfiguration STTExampleController::command_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

    for (int i = 1; i <= num_joints; ++i) {
      config.names.push_back(arm_id_ + "_joint" + to_string(i) + "/effort");
    }
    return config;
  }

  controller_interface::InterfaceConfiguration STTExampleController::state_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (int i = 1; i <= num_joints; ++i) {
      config.names.push_back(arm_id_ + "_joint" + to_string(i) + "/position");
      config.names.push_back(arm_id_ + "_joint" + to_string(i) + "/velocity");
    }
    return config;
  }

  vector<vector<double>> STTExampleController::loadTubeCoefficients() {
    static bool tube_loaded = false;
    static vector<vector<double>> cached_tubes;
    static const string tube_filename = "/home/focaslab/ros2_ws/src/STT-for-manipulators/coefficients.csv";

    if (!tube_loaded) {
      ifstream file(tube_filename);
      if (!file.is_open()) {
          RCLCPP_ERROR(get_node()->get_logger(), "Error: Could not open tube file: [%s]", tube_filename.c_str());
          tube_loaded = true;
          return cached_tubes;
      }

      string line;
      while (getline(file, line)) {
        if (line.empty()) continue;

        // Detect a new block
        if (line.find("Tube Coefficients") != string::npos) {
          cached_tubes.push_back(vector<double>());
          continue;
        }

        if (cached_tubes.empty()) continue;

        size_t delimiterPos = line.find("\",");
        if (delimiterPos != string::npos) {
          size_t valueStart = delimiterPos + 2;
          size_t nextComma = line.find(',', valueStart);

          string valueStr;
          if (nextComma != string::npos) {
            valueStr = line.substr(valueStart, nextComma - valueStart);
          } else {
            valueStr = line.substr(valueStart);
          }

          try {
            if (!valueStr.empty()) {
              cached_tubes.back().push_back(stod(valueStr));
            }
          } catch (...) {
              // Skip rows where the second column is not a valid number
          }
        }
      }
      file.close();
      tube_loaded = true;
    }

    return cached_tubes;
  }

  vector<double> STTExampleController::real_gammas(double t, const vector<double>& C_fin, int degree) {
    int dim = 7;
    vector<double> real_tubes(dim, 0.0);

    for (int i = 0; i < dim; ++i) {
      double power_val = 1.0;
      
      for (int j = 0; j <= degree; ++j) {
        int index = j + i * (degree + 1);
        real_tubes[i] += C_fin[index] * power_val;
        power_val *= t;
      }
    }

    return real_tubes;
  }

  vector<double> STTExampleController::get_tube_state(double t, const vector<vector<double>>& C_array, int dim) {
    int tube_idx = -1;

    if (t >= 0.0 && t <= 7.0) {
      tube_idx = 0;
    } else if (t > 7.0 && t <= 13.0) {
      tube_idx = 1;
    } else if (t > 13.0 && t <= 24.0) {
      tube_idx = 2;
    } else {
      return vector<double>(dim, 0.0);
    }

    if (tube_idx >= C_array.size()) {
      return vector<double>(dim, 0.0);
    }

    const vector<double>& coeffs = C_array[tube_idx];
    int current_degree = (static_cast<int>(coeffs.size()) / dim) - 1;
    return real_gammas(t, coeffs, current_degree);
  }

  Vector7d STTExampleController::compute_torque_command(const Vector7d& joint_positions_desired, const Vector7d& joint_positions_current, const Vector7d& joint_velocities_current) {

    const double kAlpha = 0.99;
    dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * joint_velocities_current;

    Vector7d taub = {10, 10 ,10 ,10, 5 ,5, 2};
    Vector7d vb = {2, 2, 2, 2, 2.5, 1.5, 1}; 

    Vector7d psi1 = {0, 0, 0, 0, 0, 0, 0};
    Vector7d psi2 = {0, 0, 0, 0, 0, 0, 0};

    double sigma = 0.1;
    double rho = 1;

    for (int i = 0; i < 7; ++i) {
      double e = (q_(i) - joint_positions_desired[i])/sigma;

      psi1[i] = pow((tanh(e)),1);
      psi2[i] = tanh((dq_filtered_[i] + vb[i]*psi1[i])/rho);
    }

    Vector7d tau_d_calculated = -taub.cwiseProduct(psi2);

    return tau_d_calculated;
  }

  controller_interface::return_type STTExampleController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
    updateJointStates();
    Vector7d q_goal = initial_q_;
    elapsed_time_ = elapsed_time_ + period.seconds();
    vector<double> delta_angle = get_tube_state(elapsed_time_, loadTubeCoefficients(), num_joints);

    if (delta_angle.empty()) {
      q_goal = q_;
      return controller_interface::return_type::OK;
    }

    for (int i = 0; i < num_joints; ++i) {
      q_goal(i) = delta_angle[i];
    }

    const double kAlpha = 0.99;
    dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;
    Vector7d tau_d_calculated = compute_torque_command(q_goal, q_, dq_filtered_);
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(tau_d_calculated(i));
    }
    return controller_interface::return_type::OK;
  }

  CallbackReturn STTExampleController::on_init() {
    try {
      auto_declare<string>("arm_id", "");
    } catch (const exception& e) {
      fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
      return CallbackReturn::ERROR;
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn STTExampleController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    dq_filtered_.setZero();

    auto parameters_client =
        make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
    parameters_client->wait_for_service();

    auto future = parameters_client->get_parameters({"robot_description"});
    auto result = future.get();
    if (!result.empty()) {
      robot_description_ = result[0].value_to_string();
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
    }

    arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

    return CallbackReturn::SUCCESS;
  }

  CallbackReturn STTExampleController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
    updateJointStates();
    dq_filtered_.setZero();
    initial_q_ = q_;
    elapsed_time_ = 0.0;

    return CallbackReturn::SUCCESS;
  }

  void STTExampleController::updateJointStates() {
    for (auto i = 0; i < num_joints; ++i) {
      const auto& position_interface = state_interfaces_.at(2 * i);
      const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

      assert(position_interface.get_interface_name() == "position");
      assert(velocity_interface.get_interface_name() == "velocity");

      q_(i) = position_interface.get_value();
      dq_(i) = velocity_interface.get_value();
    }
  }

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::STTExampleController,
                       controller_interface::ControllerInterface)
