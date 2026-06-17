// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "robot_interface.hpp"

RobotInterface::RobotInterface(const std::string& config_file) {
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
        for (auto idx : robot_cfg_->close_chain_motor_idx_) {
            auto it = std::find(robot_cfg_->urdf2motor_.begin(), robot_cfg_->urdf2motor_.end(), idx);
            if (it != robot_cfg_->urdf2motor_.end()) {
                close_chain_joint_idx_.push_back(std::distance(robot_cfg_->urdf2motor_.begin(), it));
            }
        }
        cached_ankle_action_.resize(close_chain_joint_idx_.size(), 0.0f);
        last_ankle_joint_target_.resize(close_chain_joint_idx_.size(), 0.0f);
        if (robot_node["type"]) {
            ankle_decouple_ = Decouple::create(robot_node["type"].as<std::string>());
        } else {
            ankle_decouple_ = nullptr;
        }
    } else {
        throw std::runtime_error("Robot configuration not found in " + config_file);
    }

    thread_pool_ = std::make_unique<ThreadPool>(motors_cfg_->motor_interface_.size());

    joint_q_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_vel_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_tau_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
}

void RobotInterface::setup_motors(){
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

void RobotInterface::apply_action(std::vector<float> action) {
    if(!is_init_.load()){
        return;
    }

    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            joint_q_[motor2urdf_[idx]] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
            joint_vel_[motor2urdf_[idx]] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
            joint_tau_[motor2urdf_[idx]] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];
            if (motor->get_response_count() > offline_threshold_) {
                throw std::runtime_error("Motor id " + std::to_string(motors_cfg_->motor_id_[idx]) + " offline");
            }
        });

        if (!close_chain_joint_idx_.empty() && ankle_decouple_){
            bool refresh = false;
            for (size_t i = 0; i < close_chain_joint_idx_.size(); ++i) {
                if (action[close_chain_joint_idx_[i]] != last_ankle_joint_target_[i]) {
                    refresh = true;
                }
                last_ankle_joint_target_[i] = action[close_chain_joint_idx_[i]];
            }
            Eigen::VectorXd q(2), vel(2), tau(2);
            int idx1 = close_chain_joint_idx_[0];
            int idx2 = close_chain_joint_idx_[1];
            q << joint_q_[idx1], joint_q_[idx2];
            vel << joint_vel_[idx1], joint_vel_[idx2];
            tau << joint_tau_[idx1], joint_tau_[idx2];
            ankle_decouple_->get_forwardQVT(q, vel, tau, true);
            joint_q_[idx1] = q[0];
            joint_q_[idx2] = q[1];
            joint_vel_[idx1] = vel[0];
            joint_vel_[idx2] = vel[1];
            joint_tau_[idx1] = tau[0];
            joint_tau_[idx2] = tau[1];
            if (refresh) {
                tau << robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[0]] * (action[idx1] - q[0]) + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[0]] * (0.0f - vel[0]),
                robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[1]] * (action[idx2] - q[1]) + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[1]] * (0.0f - vel[1]);
                ankle_decouple_->get_decoupleQVT(q, vel, tau, true);
                action[idx1] = static_cast<float>(q[0] + (tau[0] + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[0]] * vel[0]) / robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[0]]);
                action[idx2] = static_cast<float>(q[1] + (tau[1] + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[1]] * vel[1]) / robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[1]]);
                cached_ankle_action_[0] = action[idx1];
                cached_ankle_action_[1] = action[idx2];
            } else {
                action[idx1] = cached_ankle_action_[0];
                action[idx2] = cached_ankle_action_[1];
            }
            
            idx1 = close_chain_joint_idx_[2];
            idx2 = close_chain_joint_idx_[3];
            q << joint_q_[idx1], joint_q_[idx2];
            vel << joint_vel_[idx1], joint_vel_[idx2];
            tau << joint_tau_[idx1], joint_tau_[idx2];
            ankle_decouple_->get_forwardQVT(q, vel, tau, false);
            joint_q_[idx1] = q[0];
            joint_q_[idx2] = q[1];
            joint_vel_[idx1] = vel[0];
            joint_vel_[idx2] = vel[1];
            joint_tau_[idx1] = tau[0];
            joint_tau_[idx2] = tau[1];
            if (refresh) {
                tau << robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[2]] * (action[idx1] - q[0]) + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[2]] * (0.0f - vel[0]),
                robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[3]] * (action[idx2] - q[1]) + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[3]] * (0.0f - vel[1]);
                ankle_decouple_->get_decoupleQVT(q, vel, tau, false);
                action[idx1] = static_cast<float>(q[0] + (tau[0] + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[2]] * vel[0]) / robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[2]]);
                action[idx2] = static_cast<float>(q[1] + (tau[1] + robot_cfg_->kd_[robot_cfg_->close_chain_motor_idx_[3]] * vel[1]) / robot_cfg_->kp_[robot_cfg_->close_chain_motor_idx_[3]]);
                cached_ankle_action_[2] = action[idx1];
                cached_ankle_action_[3] = action[idx2];
            } else {
                action[idx1] = cached_ankle_action_[2];
                action[idx2] = cached_ankle_action_[3];
            }
        }
    }

    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_target_.size(); i++){
            motor_target_[i] = action[motor2urdf_[i]];
        }
    }

    motors_mit_cmd(1.0f, 1.0f);
}

void RobotInterface::reset_joints(std::vector<double> joint_default_angle) {
    if (!close_chain_joint_idx_.empty() && ankle_decouple_){
        Eigen::VectorXd q(2), vel(2), tau(2);
        int idx1 = close_chain_joint_idx_[0];
        int idx2 = close_chain_joint_idx_[1];
        q << joint_default_angle[idx1], joint_default_angle[idx2];
        ankle_decouple_->get_decoupleQVT(q, vel, tau, true);
        joint_default_angle[idx1] = q[0];
        joint_default_angle[idx2] = q[1];

        idx1 = close_chain_joint_idx_[2];
        idx2 = close_chain_joint_idx_[3];
        q << joint_default_angle[idx1], joint_default_angle[idx2];
        ankle_decouple_->get_decoupleQVT(q, vel, tau, false);
        joint_default_angle[idx1] = q[0];
        joint_default_angle[idx2] = q[1];
    }

    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_target_.size(); i++){
            motor_target_[i] = joint_default_angle[motor2urdf_[i]];
        }
    }

    motors_mit_cmd(1.0f/2.5f, 1.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    motors_mit_cmd(1.0f, 1.0f);
}

void RobotInterface::refresh_joints() {
    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            motor->refresh_motor_status();
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            joint_q_[motor2urdf_[idx]] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
            joint_vel_[motor2urdf_[idx]] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
            joint_tau_[motor2urdf_[idx]] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];
        });

        if (!close_chain_joint_idx_.empty() && ankle_decouple_) {
            Eigen::VectorXd q(2), vel(2), tau(2);
            int idx1 = close_chain_joint_idx_[0];
            int idx2 = close_chain_joint_idx_[1];
            q << joint_q_[idx1], joint_q_[idx2];
            vel << joint_vel_[idx1], joint_vel_[idx2];
            tau << joint_tau_[idx1], joint_tau_[idx2];
            ankle_decouple_->get_forwardQVT(q, vel, tau, true);
            joint_q_[idx1] = q[0];
            joint_q_[idx2] = q[1];
            joint_vel_[idx1] = vel[0];
            joint_vel_[idx2] = vel[1];
            joint_tau_[idx1] = tau[0];
            joint_tau_[idx2] = tau[1];
            
            idx1 = close_chain_joint_idx_[2];
            idx2 = close_chain_joint_idx_[3];
            q << joint_q_[idx1], joint_q_[idx2];
            vel << joint_vel_[idx1], joint_vel_[idx2];
            tau << joint_tau_[idx1], joint_tau_[idx2];
            ankle_decouple_->get_forwardQVT(q, vel, tau, false);
            joint_q_[idx1] = q[0];
            joint_q_[idx2] = q[1];
            joint_vel_[idx1] = vel[0];
            joint_vel_[idx2] = vel[1];
            joint_tau_[idx1] = tau[0];
            joint_tau_[idx2] = tau[1];
        }
    }
}

void RobotInterface::set_zeros() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->set_motor_zero();
    });
}

void RobotInterface::clear_errors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->clear_motor_error();
    });
}

void RobotInterface::init_motors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->init_motor();
    });
    is_init_.store(true);
}

void RobotInterface::deinit_motors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->deinit_motor();
    });
    is_init_.store(false);
}

void RobotInterface::motors_mit_cmd(float kp_scale, float kd_scale) {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    std::vector<std::function<void()>> tasks;
    size_t count = 0;
    for (size_t bus = 0; bus < motors_cfg_->motor_interface_.size(); ++bus) {
        const size_t num_motors = motors_cfg_->motor_num_[bus];
        const size_t start_count = count;
        if (motors_cfg_->motor_interface_type_[bus] == "canfd") {
            tasks.push_back([this, start_count, num_motors, kp_scale, kd_scale]() {
                float pos[8] = {}, vel[8] = {}, kp[8] = {}, kd[8] = {}, tau[8] = {};
                for (size_t j = 0; j < num_motors; ++j) {
                    const size_t idx = start_count + j;
                    const long int motor_id = motors_cfg_->motor_id_[idx];
                    const size_t slot = (motor_id > 0 && motor_id <= 8) ? static_cast<size_t>(motor_id - 1) : j;
                    if (slot >= 8) continue;
                    pos[slot] = motor_target_[idx] * robot_cfg_->motor_sign_[idx];
                    kp[slot] = robot_cfg_->kp_[idx] * kp_scale;
                    kd[slot] = robot_cfg_->kd_[idx] * kd_scale;
                }
                motors_[start_count]->motor_mit_cmd(pos, vel, kp, kd, tau);
            });
        } else {
            tasks.push_back([this, start_count, num_motors, kp_scale, kd_scale]() {
                for (size_t j = 0; j < num_motors; ++j) {
                    const size_t idx = start_count + j;
                    motors_[idx]->motor_mit_cmd(motor_target_[idx] * robot_cfg_->motor_sign_[idx], 0.0f,
                                                robot_cfg_->kp_[idx] * kp_scale, robot_cfg_->kd_[idx] * kd_scale, 0.0f);
                }
            });
        }
        count += num_motors;
    }
    thread_pool_->run_parallel(tasks);
}

void RobotInterface::exec_motors_parallel(const std::function<void(std::shared_ptr<MotorDriver>&, int)>& cmd_func) {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    std::vector<std::function<void()>> tasks;
    size_t count = 0;
    
    for (size_t i = 0; i < motors_cfg_->motor_interface_.size(); ++i) {
        size_t num_motors = motors_cfg_->motor_num_[i];
        size_t start_idx = count;
        tasks.push_back([this, start_idx, num_motors, cmd_func]() {
            for (size_t j = 0; j < num_motors; ++j) {
                size_t idx = start_idx + j;
                cmd_func(motors_[idx], idx); 
            }
        });
        count += num_motors;
    }
    thread_pool_->run_parallel(tasks);
}
