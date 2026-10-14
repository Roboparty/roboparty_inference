#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include "utils/close_chain_mapping.hpp"
#include "utils/thread_pool.hpp"
#include "motor_driver.hpp"
#include "imu_driver.hpp"

class RobotInterface {
   public:
    RobotInterface(const std::string& config_file, double control_dt = 0.001);
    ~RobotInterface() {
        if (is_init_.load()) {
            deinit_motors();
        }
        motors_.clear();
        imu_.reset();
    }
    struct IMUCfg{
        int imu_id_, baudrate_;
        std::string imu_type_, imu_interface_type_, imu_interface_;
    };
    struct MotorsCfg{
        int master_id_offset_;
        std::vector<std::string> motor_type_;
        std::vector<std::string> motor_interface_type_;
        std::vector<std::string> motor_interface_;
        std::vector<long int> motor_id_, motor_model_, motor_num_;
        std::vector<double> motor_zero_offset_;
    };
    struct RobotCfg{
        std::vector<long int> close_chain_motor_idx_, motor_sign_, urdf2motor_;
        std::vector<double> kp_, kd_, extrinsic_R_;
    };

    void apply_action(const std::vector<float>& p,
                      const std::vector<float>& v = {},
                      const std::vector<float>& kp = {},
                      const std::vector<float>& kd = {},
                      const std::vector<float>& tau = {});
    void init_motors();
    void deinit_motors();
    void reset_joints(std::vector<double> joint_default_angle);
    void set_zeros();
    void clear_errors();
    void read_joints();
    void read_imu();
    void refresh_joints();
    std::vector<float> get_joint_q() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors are not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_q_;
    }
    std::vector<float> get_joint_vel() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors are not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_vel_;
    }
    std::vector<float> get_joint_tau() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors are not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_tau_;
    }
    std::vector<float> get_quat() {
        if (!imu_) {
            throw std::runtime_error("IMU is not initialized");
        }
        std::unique_lock<std::mutex> lock(imu_mutex_);
        return quat_buf_;
    }
    std::vector<float> get_ang_vel() {
        if (!imu_) {
            throw std::runtime_error("IMU is not initialized");
        }
        std::unique_lock<std::mutex> lock(imu_mutex_);
        return ang_vel_buf_;
    }

    std::atomic<bool> is_init_{false};

   private:
    std::shared_ptr<IMUCfg> imu_cfg_;
    std::shared_ptr<MotorsCfg> motors_cfg_;
    std::shared_ptr<RobotCfg> robot_cfg_;
    // A motor is offline only after 100 ms without a feedback frame.
    int offline_threshold_cycles_ = 100;
    std::shared_ptr<IMUDriver> imu_;
    std::shared_ptr<Decouple> ankle_decouple_;
    Eigen::Matrix3f extrinsic_R_mat_ = Eigen::Matrix3f::Identity();
    Eigen::Quaternionf extrinsic_q_inv_ = Eigen::Quaternionf::Identity();
    std::vector<float> quat_buf_{0.f, 0.f, 0.f, 0.f};
    std::vector<float> ang_vel_buf_{0.f, 0.f, 0.f};
    std::vector<std::shared_ptr<MotorDriver>> motors_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::vector<float> last_ankle_joint_target_;
    std::vector<size_t> motor_bus_offsets_;

    std::mutex reset_mutex_, lifecycle_mutex_, command_mutex_, motors_mutex_, joint_mutex_, imu_mutex_;
    std::vector<float> joint_q_, joint_vel_, joint_tau_;
    std::vector<float> motor_pos_target_, motor_vel_target_, motor_kp_target_, motor_kd_target_, motor_tau_target_;
    // True when joint_q_/joint_vel_/joint_tau_ contain feedback from the motors.
    std::atomic<bool> joint_cache_valid_{false};
    std::vector<int> close_chain_joint_idx_, motor2urdf_;

    void setup_motors();
    void setup_imu();

    template <typename F>
    void exec_motors_parallel(F&& cmd_func) {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        thread_pool_->run_parallel(motor_bus_offsets_.size() - 1,
            [this, &cmd_func](size_t bus) {
                for (size_t idx = motor_bus_offsets_[bus];
                     idx < motor_bus_offsets_[bus + 1]; ++idx) {
                    cmd_func(motors_[idx], static_cast<int>(idx));
                }
            });
    }
    void throw_if_motors_offline() const;
    void motors_mit_cmd();
    void forward_close_chain();
};
