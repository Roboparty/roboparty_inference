// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "robot_interface.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

static constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);

static float wrap_joint(float position) {
    float wrapped = std::remainder(position, kTwoPi);
    return wrapped >= static_cast<float>(M_PI) ? wrapped - kTwoPi : wrapped;
}

static float unwrap_target(float target, float current_position) {
    return target + kTwoPi * std::round((current_position - target) / kTwoPi);
}

RobotInterface::RobotInterface(const std::string& config_file, double control_dt) {
    if (control_dt <= 0.0) {
        throw std::invalid_argument("control_dt must be greater than zero");
    }
    offline_threshold_cycles_ = std::max(1, static_cast<int>(std::ceil(0.1 / control_dt)));
    YAML::Node config = YAML::LoadFile(config_file);

    imu_cfg_ = std::make_shared<IMUCfg>();
    if (config["imu"]) {
        YAML::Node imu_node = config["imu"];
        if (imu_node["imu_id"]) imu_cfg_->imu_id_ = imu_node["imu_id"].as<int>();
        if (imu_node["baudrate"]) imu_cfg_->baudrate_ = imu_node["baudrate"].as<int>();
        if (imu_node["imu_type"]) imu_cfg_->imu_type_ = imu_node["imu_type"].as<std::string>();
        if (imu_node["imu_interface_type"]) imu_cfg_->imu_interface_type_ = imu_node["imu_interface_type"].as<std::string>();
        if (imu_node["imu_interface"]) imu_cfg_->imu_interface_ = imu_node["imu_interface"].as<std::string>();
        setup_imu();
    }

    motors_cfg_ = std::make_shared<MotorsCfg>();
    if (config["motors"]) {
        YAML::Node motors_node = config["motors"];
        if (motors_node["motor_zero_offset"]) motors_cfg_->motor_zero_offset_ = motors_node["motor_zero_offset"].as<std::vector<double>>();
        if (motors_node["master_id_offset"]) motors_cfg_->master_id_offset_ = motors_node["master_id_offset"].as<int>();
        if (motors_node["motor_type"]) motors_cfg_->motor_type_ = motors_node["motor_type"].as<std::vector<std::string>>();
        if (motors_node["motor_interface_type"]) motors_cfg_->motor_interface_type_ = motors_node["motor_interface_type"].as<std::vector<std::string>>();
        if (motors_node["motor_interface"]) motors_cfg_->motor_interface_ = motors_node["motor_interface"].as<std::vector<std::string>>();
        if (motors_node["motor_id"]) motors_cfg_->motor_id_ = motors_node["motor_id"].as<std::vector<long int>>();
        if (motors_node["motor_model"]) motors_cfg_->motor_model_ = motors_node["motor_model"].as<std::vector<long int>>();
        if (motors_node["motor_num"]) motors_cfg_->motor_num_ = motors_node["motor_num"].as<std::vector<long int>>();
        setup_motors();
    } else {
        throw std::runtime_error("Motors configuration not found in " + config_file);
    }

    robot_cfg_ = std::make_shared<RobotCfg>();
    if (config["robot"]) {
        YAML::Node robot_node = config["robot"];
        if (robot_node["kp"]) robot_cfg_->kp_ = robot_node["kp"].as<std::vector<double>>();
        if (robot_node["kd"]) robot_cfg_->kd_ = robot_node["kd"].as<std::vector<double>>();
        if (robot_node["close_chain_motor_idx"]) robot_cfg_->close_chain_motor_idx_ = robot_node["close_chain_motor_idx"].as<std::vector<long int>>();
        if (robot_node["motor_sign"]) robot_cfg_->motor_sign_ = robot_node["motor_sign"].as<std::vector<long int>>();
        if (robot_node["urdf2motor"]) robot_cfg_->urdf2motor_ = robot_node["urdf2motor"].as<std::vector<long int>>();
        motor2urdf_ = std::vector<int>(motors_cfg_->motor_id_.size(), -1);
        for (size_t i = 0; i < robot_cfg_->urdf2motor_.size(); ++i) {
            motor2urdf_[robot_cfg_->urdf2motor_[i]] = i;
        }
        if (robot_node["extrinsic_R"]) {
            robot_cfg_->extrinsic_R_ = robot_node["extrinsic_R"].as<std::vector<double>>();
            if (robot_cfg_->extrinsic_R_.size() == 9) {
                // Row-major: [r00, r01, r02, r10, r11, r12, r20, r21, r22]
                extrinsic_R_mat_ << robot_cfg_->extrinsic_R_[0], robot_cfg_->extrinsic_R_[1], robot_cfg_->extrinsic_R_[2],
                                    robot_cfg_->extrinsic_R_[3], robot_cfg_->extrinsic_R_[4], robot_cfg_->extrinsic_R_[5],
                                    robot_cfg_->extrinsic_R_[6], robot_cfg_->extrinsic_R_[7], robot_cfg_->extrinsic_R_[8];
                Eigen::Quaternionf q_R(extrinsic_R_mat_);  // quaternion of R (Body->IMU)
                extrinsic_q_inv_ = q_R.inverse();           // we need R_inv for quaternion transform
            }
        }
        for (size_t ankle_idx = 0; ankle_idx < robot_cfg_->close_chain_motor_idx_.size(); ++ankle_idx) {
            const long int idx = robot_cfg_->close_chain_motor_idx_[ankle_idx];
            auto it = std::find(robot_cfg_->urdf2motor_.begin(), robot_cfg_->urdf2motor_.end(), idx);
            if (it != robot_cfg_->urdf2motor_.end()) {
                close_chain_joint_idx_.push_back(std::distance(robot_cfg_->urdf2motor_.begin(), it));
            }
        }
        last_ankle_joint_target_.resize(close_chain_joint_idx_.size(), 0.0f);
        if (robot_node["type"]) {
            ankle_decouple_ = Decouple::create(robot_node["type"].as<std::string>());
        } else {
            ankle_decouple_ = nullptr;
        }
    } else {
        throw std::runtime_error("Robot configuration not found in " + config_file);
    }

    thread_pool_ = std::make_unique<ThreadPool>(motors_cfg_->motor_interface_.size(), 46);

    joint_q_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_vel_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_tau_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_pos_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_vel_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_kp_target_  = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_kd_target_  = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_tau_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
}

void RobotInterface::setup_motors(){
    const size_t bus_count = motors_cfg_->motor_interface_.size();
    motor_bus_offsets_.assign(bus_count + 1, 0);
    for (size_t bus = 0; bus < bus_count; ++bus) {
        motor_bus_offsets_[bus + 1] = motor_bus_offsets_[bus] +
                                      static_cast<size_t>(motors_cfg_->motor_num_[bus]);
    }

    size_t count = 0;
    motors_.resize(motors_cfg_->motor_id_.size());
    for (size_t i = 0; i < motors_cfg_->motor_interface_.size(); ++i){
        for (size_t j = 0; j < motors_cfg_->motor_num_[i]; ++j){
            motors_[count] = MotorDriver::create_motor(motors_cfg_->motor_id_[count], motors_cfg_->motor_interface_type_[i], motors_cfg_->motor_interface_[i], motors_cfg_->motor_type_[i], motors_cfg_->motor_model_[count], motors_cfg_->master_id_offset_, motors_cfg_->motor_zero_offset_[count]);
            count += 1;
        }
    }
}

void RobotInterface::setup_imu(){
    imu_ = IMUDriver::create_imu(imu_cfg_->imu_id_, imu_cfg_->imu_interface_type_, imu_cfg_->imu_interface_, imu_cfg_->imu_type_, imu_cfg_->baudrate_);
}

void RobotInterface::forward_close_chain() {
    Eigen::Vector2d q, vel, tau;
    for (size_t pair = 0; pair < 2; ++pair) {
        const bool left = (pair == 0);
        int idx1 = close_chain_joint_idx_[pair * 2];
        int idx2 = close_chain_joint_idx_[pair * 2 + 1];
        q << joint_q_[idx1], joint_q_[idx2];
        vel << joint_vel_[idx1], joint_vel_[idx2];
        tau << joint_tau_[idx1], joint_tau_[idx2];
        ankle_decouple_->get_forwardQVT(q, vel, tau, left);
        joint_q_[idx1]   = q[0];
        joint_q_[idx2]   = q[1];
        joint_vel_[idx1] = vel[0];
        joint_vel_[idx2] = vel[1];
        joint_tau_[idx1] = tau[0];
        joint_tau_[idx2] = tau[1];
    }
}

void RobotInterface::read_joints() {
    if (!is_init_.load()) {
        throw std::runtime_error("Motors are not initialized");
    }
    std::unique_lock<std::mutex> lock(joint_mutex_);
    {
        std::unique_lock<std::mutex> motor_lock(motors_mutex_);
        for (size_t idx = 0; idx < motors_.size(); ++idx) {
            const auto& motor = motors_[idx];
            const float sign = static_cast<float>(robot_cfg_->motor_sign_[idx]);
            joint_q_[motor2urdf_[idx]] = wrap_joint(motor->get_motor_pos() * sign);
            joint_vel_[motor2urdf_[idx]] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
            joint_tau_[motor2urdf_[idx]] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];
        }
    }
    throw_if_motors_offline();

    if (!close_chain_joint_idx_.empty() && ankle_decouple_) {
        forward_close_chain();
    }
}

void RobotInterface::read_imu() {
    if (!imu_) {
        throw std::runtime_error("IMU is not initialized");
    }

    std::unique_lock<std::mutex> lock(imu_mutex_);
    const auto raw_quat = imu_->get_quat();          // w, x, y, z
    const auto raw_ang_vel = imu_->get_ang_vel();  // in IMU frame
    Eigen::Quaternionf q_body =
        Eigen::Quaternionf(raw_quat[0], raw_quat[1], raw_quat[2], raw_quat[3]) * extrinsic_q_inv_;
    q_body.normalize();
    Eigen::Map<const Eigen::Vector3f> omega_imu(raw_ang_vel.data());
    const Eigen::Vector3f omega_body = extrinsic_R_mat_ * omega_imu;

    quat_buf_[0] = q_body.w();
    quat_buf_[1] = q_body.x();
    quat_buf_[2] = q_body.y();
    quat_buf_[3] = q_body.z();
    Eigen::Map<Eigen::Vector3f>(ang_vel_buf_.data()) = omega_body;
}

void RobotInterface::apply_action(const std::vector<float>& p,
                                  const std::vector<float>& v,
                                  const std::vector<float>& kp,
                                  const std::vector<float>& kd,
                                  const std::vector<float>& tau) {
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    if(!is_init_.load()){
        return;
    }
    const bool use_close_chain_position = !close_chain_joint_idx_.empty() && ankle_decouple_;
    control_started_ = true;

    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        {
            std::unique_lock<std::mutex> motor_lock(motors_mutex_);
            for (size_t idx = 0; idx < motors_.size(); ++idx) {
                const auto& motor = motors_[idx];
                const float motor_pos = motor->get_motor_pos();
                const float motor_vel = motor->get_motor_spd();
                const float motor_tau = motor->get_motor_current();
                const float sign = static_cast<float>(robot_cfg_->motor_sign_[idx]);
                joint_q_[motor2urdf_[idx]] = wrap_joint(motor_pos * sign);
                joint_vel_[motor2urdf_[idx]] = motor_vel * robot_cfg_->motor_sign_[idx];
                joint_tau_[motor2urdf_[idx]] = motor_tau * robot_cfg_->motor_sign_[idx];
            }
        }
        throw_if_motors_offline();

        if (use_close_chain_position){
            auto kp_cc = [&](size_t i) -> double {
                return kp.empty() ? robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[i]]
                                  : static_cast<double>(kp[close_chain_joint_idx_[i]]);
            };
            auto kd_cc = [&](size_t i) -> double {
                return kd.empty() ? robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[i]]
                                  : static_cast<double>(kd[close_chain_joint_idx_[i]]);
            };
            auto vel_target_cc = [&](size_t i) -> double {
                return v.empty() ? 0.0 : static_cast<double>(v[close_chain_joint_idx_[i]]);
            };
            auto tau_ff_cc = [&](size_t i) -> double {
                return tau.empty() ? 0.0 : static_cast<double>(tau[close_chain_joint_idx_[i]]);
            };

            std::array<float, 4> motor_pos{};
            std::array<float, 4> motor_vel{};
            for (size_t i = 0; i < motor_pos.size(); ++i) {
                const int joint_idx = close_chain_joint_idx_[i];
                motor_pos[i] = joint_q_[joint_idx];
                motor_vel[i] = joint_vel_[joint_idx];
            }

            forward_close_chain();

            Eigen::Vector2d q, vel, tau_cc;
            for (size_t pair = 0; pair < 2; ++pair) {
                const bool left = (pair == 0);
                const size_t off = pair * 2;
                const int idx1 = close_chain_joint_idx_[off];
                const int idx2 = close_chain_joint_idx_[off + 1];
                const double motor_kp1 = kp_cc(off);
                const double motor_kp2 = kp_cc(off + 1);
                const double motor_kd1 = kd_cc(off);
                const double motor_kd2 = kd_cc(off + 1);

                q << joint_q_[idx1], joint_q_[idx2];
                vel << joint_vel_[idx1], joint_vel_[idx2];
                tau_cc << motor_kp1 * (p[idx1] - q[0]) + motor_kd1 * (vel_target_cc(off)     - vel[0]) + tau_ff_cc(off),
                          motor_kp2 * (p[idx2] - q[1]) + motor_kd2 * (vel_target_cc(off + 1) - vel[1]) + tau_ff_cc(off + 1);
                ankle_decouple_->get_decoupleQVT(q, vel, tau_cc, left);

                const double theta1 = motor_pos[off];
                const double theta2 = motor_pos[off + 1];
                const double theta_vel1 = motor_vel[off];
                const double theta_vel2 = motor_vel[off + 1];

                last_ankle_joint_target_[off] = static_cast<float>(
                    theta1 + (tau_cc[0] + motor_kd1 * theta_vel1) / motor_kp1);
                last_ankle_joint_target_[off + 1] = static_cast<float>(
                    theta2 + (tau_cc[1] + motor_kd2 * theta_vel2) / motor_kp2);
            }
        }
    }

    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_pos_target_.size(); i++){
            const size_t ji = motor2urdf_[i];
            motor_pos_target_[i] = p[ji];
            motor_vel_target_[i] = v.empty() ? 0.0f : v[ji];
            motor_kp_target_[i]  = kp.empty() ? static_cast<float>(robot_cfg_->kp_[i]) : kp[ji];
            motor_kd_target_[i]  = kd.empty() ? static_cast<float>(robot_cfg_->kd_[i]) : kd[ji];
            motor_tau_target_[i] = tau.empty() ? 0.0f : tau[ji];
        }
        if (use_close_chain_position) {
            for (size_t ankle_idx = 0; ankle_idx < last_ankle_joint_target_.size(); ++ankle_idx) {
                const size_t motor_idx = robot_cfg_->close_chain_motor_idx_[ankle_idx];
                motor_pos_target_[motor_idx] = last_ankle_joint_target_[ankle_idx];
                motor_vel_target_[motor_idx] = 0.0f;
                motor_tau_target_[motor_idx] = 0.0f;
            }
        }
    }

    motors_mit_cmd();
}

void RobotInterface::reset_joints(std::vector<double> joint_default_angle) {
    std::unique_lock<std::mutex> reset_lock(reset_mutex_, std::try_to_lock);
    if (!reset_lock.owns_lock()) {
        throw std::runtime_error("Joint reset is already in progress");
    }
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    if (joint_default_angle.size() != motors_.size() ||
        !std::all_of(joint_default_angle.begin(), joint_default_angle.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Reset targets must contain one finite angle per motor");
    }

    constexpr int reset_duration_ms = 4000;
    constexpr int reset_period_ms = 20;
    constexpr int reset_steps = reset_duration_ms / reset_period_ms;
    const auto reset_period = std::chrono::milliseconds(reset_period_ms);

    reset_completed_.store(false);
    if (!control_started_) {
        refresh_joints_impl();
    } else {
        read_joints();
    }
    const std::vector<float> start_joint_q = get_joint_q();
    control_started_ = true;
    auto next_frame = std::chrono::steady_clock::now();

    for (int step = 0; step <= reset_steps; ++step) {
        if (!is_init_.load()) {
            throw std::runtime_error("Joint reset interrupted by deinitialization");
        }
        const double phase = static_cast<double>(step) / reset_steps;
        const double blend = phase * phase * (3.0 - 2.0 * phase);
        std::vector<double> joint_target(joint_default_angle.size());
        for (size_t i = 0; i < joint_target.size(); ++i) {
            joint_target[i] = start_joint_q[i] +
                              blend * (joint_default_angle[i] - start_joint_q[i]);
        }

        if (!close_chain_joint_idx_.empty() && ankle_decouple_) {
            Eigen::Vector2d q, vel = Eigen::Vector2d::Zero(), tau = Eigen::Vector2d::Zero();
            for (size_t pair = 0; pair < 2; ++pair) {
                const bool left = (pair == 0);
                int idx1 = close_chain_joint_idx_[pair * 2];
                int idx2 = close_chain_joint_idx_[pair * 2 + 1];
                q << joint_target[idx1], joint_target[idx2];
                ankle_decouple_->get_decoupleQVT(q, vel, tau, left);
                joint_target[idx1] = q[0];
                joint_target[idx2] = q[1];
            }
        }

        {
            std::unique_lock<std::mutex> lock(motors_mutex_);
            for (size_t i = 0; i < motor_pos_target_.size(); i++){
                motor_pos_target_[i] = static_cast<float>(joint_target[motor2urdf_[i]]);
                motor_vel_target_[i] = 0.0f;
                motor_kp_target_[i]  = static_cast<float>(robot_cfg_->kp_[i]) * (1.0f / 2.5f);
                motor_kd_target_[i]  = static_cast<float>(robot_cfg_->kd_[i]);
                motor_tau_target_[i] = 0.0f;
            }
        }

        motors_mit_cmd();
        if (step < reset_steps) {
            next_frame += reset_period;
            std::this_thread::sleep_until(next_frame);
        }
    }

    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_kp_target_.size(); i++){
            motor_kp_target_[i] = static_cast<float>(robot_cfg_->kp_[i]);
        }
    }
    motors_mit_cmd();
    read_joints();
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (is_init_.load()) {
        reset_completed_.store(true);
    }
}

void RobotInterface::refresh_joints() {
    std::unique_lock<std::mutex> command_lock(command_mutex_, std::try_to_lock);
    if (!command_lock.owns_lock() || control_started_) {
        throw std::runtime_error("Motor refresh is only allowed before control starts");
    }
    refresh_joints_impl();
}

void RobotInterface::refresh_joints_impl() {
    if (!is_init_.load()) {
        throw std::runtime_error("Motors are not initialized");
    }
    exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int) {
        // RP refresh can enable a motor, so cancellation must also gate refresh sends.
        if (is_init_.load()) {
            motor->refresh_motor_status();
        }
    });
    for (int step = 0; step < 50; ++step) {
        if (!is_init_.load()) {
            throw std::runtime_error("Motor refresh interrupted by deinitialization");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    read_joints();
}

void RobotInterface::set_zeros() {
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    if (!is_init_.load()) {
        throw std::runtime_error("Motors are not initialized");
    }
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->set_motor_zero();
    });
}

void RobotInterface::clear_errors() {
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->clear_motor_error();
    });
}

void RobotInterface::init_motors() {
    std::unique_lock<std::mutex> reset_lock(reset_mutex_);
    std::unique_lock<std::mutex> command_lock(command_mutex_);
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (is_init_.load()) {
        throw std::runtime_error("Motors are already initialized");
    }
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->init_motor();
    });
    control_started_ = false;
    reset_completed_.store(false);
    is_init_.store(true);
}

void RobotInterface::deinit_motors() {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (!is_init_.exchange(false)) {
        throw std::runtime_error("Motors are already deinitialized");
    }
    reset_completed_.store(false);
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int) {
        motor->deinit_motor();
    });
}

void RobotInterface::motors_mit_cmd() {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    if (!is_init_.load()) {
        return;
    }
    for (size_t bus = 0; bus + 1 < motor_bus_offsets_.size(); ++bus) {
        const size_t start_count = motor_bus_offsets_[bus];
        const size_t end_count = motor_bus_offsets_[bus + 1];
        if (motors_cfg_->motor_interface_type_[bus] == "canfd") {
            float pos[8] = {}, vel[8] = {}, kp[8] = {}, kd[8] = {}, tau[8] = {};
            for (size_t idx = start_count; idx < end_count; ++idx) {
                const long int motor_id = motors_cfg_->motor_id_[idx];
                const size_t slot = (motor_id > 0 && motor_id <= 8)
                    ? static_cast<size_t>(motor_id - 1)
                    : idx - start_count;
                if (slot >= 8) continue;
                const float sign = static_cast<float>(robot_cfg_->motor_sign_[idx]);
                const float motor_target = motor_pos_target_[idx] * sign;
                pos[slot] = unwrap_target(motor_target, motors_[idx]->get_motor_pos());
                vel[slot] = motor_vel_target_[idx] * sign;
                kp[slot]  = motor_kp_target_[idx];
                kd[slot]  = motor_kd_target_[idx];
                tau[slot] = motor_tau_target_[idx] * sign;
            }
            motors_[start_count]->motor_mit_cmd(pos, vel, kp, kd, tau);
        } else {
            for (size_t idx = start_count; idx < end_count; ++idx) {
                const float sign = static_cast<float>(robot_cfg_->motor_sign_[idx]);
                const float motor_target = unwrap_target(
                    motor_pos_target_[idx] * sign, motors_[idx]->get_motor_pos());
                motors_[idx]->motor_mit_cmd(motor_target,
                                             motor_vel_target_[idx] * sign,
                                             motor_kp_target_[idx],
                                             motor_kd_target_[idx],
                                             motor_tau_target_[idx] * sign);
            }
        }
    }
}

void RobotInterface::throw_if_motors_offline() const {
    std::string offline_motors;
    for (size_t idx = 0; idx < motors_.size(); ++idx) {
        const int response_count = motors_[idx]->get_response_count();
        if (response_count < offline_threshold_cycles_) {
            continue;
        }
        if (!offline_motors.empty()) {
            offline_motors += ", ";
        }
        offline_motors += motors_[idx]->get_can_name() + "/id=" +
                          std::to_string(motors_cfg_->motor_id_[idx]) +
                          "(count=" + std::to_string(response_count) + ")";
    }
    if (!offline_motors.empty()) {
        throw std::runtime_error("Motors offline: " + offline_motors);
    }
}
