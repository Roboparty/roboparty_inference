// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "inference_node.hpp"

#include <cstdlib>
#include <cstring>

void InferenceNode::update_obs_history(std::vector<float>& history,
                                       const std::vector<float>& obs,
                                       int obs_num, int frame_stack,
                                       bool is_first_frame) {
    if (is_first_frame) {
        for (int frame = 0; frame < frame_stack; frame++) {
            std::copy(obs.begin(), obs.end(), history.begin() + frame * obs_num);
        }
        return;
    }
    std::move(history.begin() + obs_num,
              history.begin() + frame_stack * obs_num,
              history.begin());
    std::copy(obs.begin(), obs.end(), history.begin() + (frame_stack - 1) * obs_num);
}

ObsStackOrder InferenceNode::parse_obs_stack_order(const std::string& stack_order_name) {
    if (stack_order_name == "frame_major") {
        return ObsStackOrder::FrameMajor;
    }
    if (stack_order_name == "obs_major") {
        return ObsStackOrder::ObsMajor;
    }
    throw std::runtime_error("Unsupported obs stack order: " + stack_order_name);
}

void InferenceNode::update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                                       int obs_num, int frame_stack, ObsStackOrder stack_order,
                                       const std::vector<int>& field_sizes, bool is_first_frame) {
    if (stack_order == ObsStackOrder::FrameMajor) {
        update_obs_history(input_buffer, obs, obs_num, frame_stack, is_first_frame);
        return;
    }

    int input_offset = 0;
    int obs_offset = 0;

    for (const int field_size : field_sizes) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                          input_buffer.begin() + input_offset + frame * field_size);
            }
        } else {
            std::move(input_buffer.begin() + input_offset + field_size,
                      input_buffer.begin() + input_offset + frame_stack * field_size,
                      input_buffer.begin() + input_offset);
            std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                      input_buffer.begin() + input_offset + (frame_stack - 1) * field_size);
        }
        input_offset += field_size * frame_stack;
        obs_offset += field_size;
    }
}

void InferenceNode::gather_sparse_obs_history(
    std::vector<float>& input_buffer,
    const std::vector<float>& obs_history,
    const std::vector<ObsHistorySlice>& gather_plan) {
    auto output = input_buffer.begin();
    for (const ObsHistorySlice& slice : gather_plan) {
        output = std::copy_n(
            obs_history.begin() + slice.history_offset, slice.size, output);
    }
}

void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size) {
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    ctx->num_inputs = ctx->session->GetInputCount();
    if (ctx->num_inputs != 1) {
        throw std::runtime_error("Only single-input ONNX models are supported: " + model_path);
    }
    ctx->input_names.resize(ctx->num_inputs);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
    }

    size_t model_input_size = 1;
    for (size_t i = 0; i < ctx->input_shape.size(); i++) {
        model_input_size *= static_cast<size_t>(ctx->input_shape[i]);
    }
    if (model_input_size != static_cast<size_t>(input_size)) {
        throw std::runtime_error(
            "ONNX input size mismatch for " + model_path + ": model expects " +
            std::to_string(model_input_size) + " values, but config provides " + std::to_string(input_size));
    }
    ctx->input_buffer.resize(input_size);

    ctx->num_outputs = ctx->session->GetOutputCount();
    ctx->output_names.resize(ctx->num_outputs);
    ctx->output_buffer.resize(joint_num_);

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->output_shape[0] == -1) ctx->output_shape[0] = 1;
        if (ctx->output_shape[1] == -1) ctx->output_shape[1] = joint_num_;
    }

    ctx->input_names_raw = std::vector<const char *>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char *>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(), ctx->input_shape.data(), ctx->input_shape.size()));
        
    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(), ctx->output_shape.data(), ctx->output_shape.size()));
}

void InferenceNode::reset_runtime_state() {
    is_running_.store(false);
    std::unique_lock<std::mutex> mode_lock(mode_mutex_);
    std::unique_lock<std::mutex> control_lock(control_mutex_);
    is_interrupt_.store(false);
    interrupt_topic_received_.store(false);
    interrupt_feedback_initialized_ = false;
    is_motion_policy_.store(false);
    active_policy_idx_ = 0;
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(perception_mutex_);
        std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        policy_transition_elapsed_ = policy_transition_duration_;
        for (int i = 0; i < joint_num_; i++) {
            act_[i] = static_cast<float>(joint_default_angle_[i]);
            last_act_[i] = static_cast<float>(joint_default_angle_[i]);
        }
    }
    if (supports_interrupt()) {
        if (joint_default_angle_.size() < interrupt_action_.size()) {
            throw std::runtime_error("joint_default_angle is smaller than interrupt_action");
        }
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        const size_t offset = joint_default_angle_.size() - interrupt_action_.size();
        for (size_t i = 0; i < interrupt_action_.size(); i++) {
            interrupt_action_[i] = static_cast<float>(joint_default_angle_[offset + i]);
        }
    }
    for (PolicyRuntime& policy : policies_) {
        reset_policy_runtime(policy);
    }
    request_depth_history_reset();
}

void InferenceNode::request_depth_history_reset() {
    if (!use_depth_ || !clear_depth_history_client_) {
        return;
    }
    if (!clear_depth_history_client_->service_is_ready()) {
        return;
    }
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    clear_depth_history_client_->async_send_request(request);
}

InferenceNode::PolicyRuntime& InferenceNode::active_policy() {
    return policies_[active_policy_idx_];
}

void InferenceNode::switch_policy(int policy_idx) {
    std::unique_lock<std::mutex> mode_lock(mode_mutex_);
    std::unique_lock<std::mutex> control_lock(control_mutex_);
    if (policy_idx < 0 || static_cast<size_t>(policy_idx) >= policies_.size()) {
        throw std::out_of_range("Policy index out of range: " + std::to_string(policy_idx));
    }
    if (policy_idx == active_policy_idx_) {
        return;
    }
    const auto& previous = active_policy();
    const int previous_idx = active_policy_idx_;
    active_policy_idx_ = policy_idx;
    try {
        auto& policy = active_policy();
        reset_policy_runtime(policy);
        std::copy(previous.ctx->output_buffer.begin(), previous.ctx->output_buffer.end(),
                  policy.ctx->output_buffer.begin());
        if (!previous.is_first_frame) {
            update_obs_segments(policy.obs_segments, policy.obs_layout);
            flatten_obs_segments(policy.obs_segments, policy.obs.begin());
            std::transform(policy.obs.begin(), policy.obs.end(), policy.obs.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });
            const auto current_obs = policy.obs;
            // Match fields by identity; pad unavailable older frames with the oldest sample.
            for (int frame = 0; frame < policy.frame_stack; ++frame) {
                policy.obs = current_obs;
                const int old_frame = std::max(0, previous.frame_stack - policy.frame_stack + frame);
                size_t offset = 0;
                for (const auto& field : policy.obs_layout) {
                    size_t old_offset = 0;
                    for (const auto& old_field : previous.obs_layout) {
                        if (field.source == old_field.source && field.size == old_field.size &&
                            field.name != "motion_command" && field.name != "latent") {
                            const bool sparse = !previous.history_gather_plan.empty();
                            const auto& history = sparse ? previous.obs_history : previous.ctx->input_buffer;
                            const size_t source_offset = sparse || previous.stack_order == ObsStackOrder::FrameMajor
                                ? old_frame * previous.obs_num + old_offset
                                : old_offset * previous.frame_stack + old_frame * old_field.size;
                            std::copy_n(history.begin() + source_offset, field.size, policy.obs.begin() + offset);
                            break;
                        }
                        old_offset += old_field.size;
                    }
                    offset += field.size;
                }
                if (!policy.history_gather_plan.empty()) {
                    update_obs_history(policy.obs_history, policy.obs, policy.obs_num,
                                       policy.frame_stack, frame == 0);
                } else {
                    update_stacked_obs(policy.ctx->input_buffer, policy.obs, policy.obs_num,
                                       policy.frame_stack, policy.stack_order,
                                       policy.obs_layout_sizes, frame == 0);
                }
            }
            if (!policy.history_gather_plan.empty()) {
                gather_sparse_obs_history(policy.ctx->input_buffer, policy.obs_history,
                                          policy.history_gather_plan);
            }
            policy.is_first_frame = false;
        }
    } catch (...) {
        active_policy_idx_ = previous_idx;
        throw;
    }
    is_motion_policy_.store(active_policy().motion_loader != nullptr);
    {
        std::unique_lock<std::mutex> action_lock(act_mutex_);
        act_ = last_act_;
        policy_transition_elapsed_ = 0.0f;
    }
    RCLCPP_INFO(this->get_logger(), "Policy enabled: %s", active_policy().name.c_str());
}

void InferenceNode::initialize_runtime_state() {
    active_policy_idx_ = 0;

    joint_state_msg_.name.resize(joint_num_);
    joint_state_msg_.position.assign(joint_num_, 0.0f);
    joint_state_msg_.velocity.assign(joint_num_, 0.0f);
    joint_state_msg_.effort.assign(joint_num_, 0.0f);
    action_msg_.name.resize(joint_num_);
    action_msg_.position.assign(joint_num_, 0.0f);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.name[i] = "joint_" + std::to_string(i + 1);
        action_msg_.name[i] = "action_" + std::to_string(i + 1);
    }

    cmd_vel_.assign(3, 0.0f);
    act_.assign(joint_num_, 0.0f);
    last_act_.assign(joint_num_, 0.0f);
    joint_pos_buffer_.assign(joint_num_, 0.0f);
    joint_vel_buffer_.assign(joint_num_, 0.0f);
    joint_torques_buffer_.assign(joint_num_, 0.0f);
    quat_buffer_.assign(4, 0.0f);
    ang_vel_buffer_.assign(3, 0.0f);
    if (has_obs_source("perception")) {
        perception_obs_buffer_.assign(perception_obs_num_, 0.0f);
    } else {
        perception_obs_buffer_.clear();
    }
    if (has_obs_source("interrupt")) {
        interrupt_action_.assign(10, 0.0f);
    } else {
        interrupt_action_.clear();
    }
}

bool InferenceNode::has_motion_policy() const {
    return !motion_policy_indices_.empty();
}

bool InferenceNode::has_latent_policy() const {
    return std::any_of(policies_.begin(), policies_.end(), [](const PolicyRuntime& policy) {
        return policy.latent_loader != nullptr;
    });
}

bool InferenceNode::supports_interrupt() const {
    return !interrupt_action_.empty();
}

void InferenceNode::reset_policy_runtime(PolicyRuntime& policy) {
    std::fill(policy.obs.begin(), policy.obs.end(), 0.0f);
    for (auto& segment : policy.obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    if (policy.ctx) {
        std::fill(policy.ctx->input_buffer.begin(), policy.ctx->input_buffer.end(), 0.0f);
        std::fill(policy.ctx->output_buffer.begin(), policy.ctx->output_buffer.end(), 0.0f);
    }
    policy.motion_frame = 0;
    if (policy.latent_loader) {
        policy.latent_loader->reset();
    }
    policy.is_first_frame = true;
}

void InferenceNode::apply_action() {
    std::unique_lock<std::mutex> control_lock(control_mutex_);
    if(!is_running_.load()){
        return;
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        float alpha = act_alpha_;
        if (policy_transition_elapsed_ < policy_transition_duration_) {
            const float progress = policy_transition_elapsed_ / policy_transition_duration_;
            const float weight = progress * progress * (3.0f - 2.0f * progress);
            const float initial_alpha = std::min(act_alpha_, 0.02f);
            alpha = initial_alpha + weight * (act_alpha_ - initial_alpha);
            policy_transition_elapsed_ = std::min(
                policy_transition_elapsed_ + dt_, policy_transition_duration_);
        }
        for (size_t i = 0; i < act_.size(); i++) {
            last_act_[i] = alpha * act_[i] + (1 - alpha) * last_act_[i];
        }
    }
    robot_->apply_action(last_act_);
}

bool InferenceNode::bind_thread_cpu(const char* cpu_env, int default_cpu) {
    int cpu = default_cpu;
    if (const char* value = std::getenv(cpu_env)) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (*value >= '0' && *value <= '9' && *end == '\0' && parsed < CPU_SETSIZE) {
            cpu = static_cast<int>(parsed);
        } else {
            RCLCPP_WARN(get_logger(), "Invalid %s='%s'; using CPU%d", cpu_env, value, cpu);
        }
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (error != 0) {
        RCLCPP_ERROR(get_logger(), "%s CPU%d affinity failed: %s", cpu_env, cpu, std::strerror(error));
    }
    return error == 0;
}

void InferenceNode::control() {
    pthread_setname_np(pthread_self(), "control");
    if (!bind_thread_cpu("ROBOT_CONTROL_CPU", 5)) {
        rclcpp::shutdown();
        return;
    }
    struct sched_param sp{}; sp.sched_priority = 45;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for control thread");
        rclcpp::shutdown();
        return;
    }
    const auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000000));
    auto next_release = std::chrono::steady_clock::now();
    while(rclcpp::ok()){
        next_release += period;
        try {
            apply_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in control thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        auto loop_end = std::chrono::steady_clock::now();
        if (loop_end > next_release) {
            const auto missed_periods = (loop_end - next_release) / period;
            next_release += period * missed_periods;
            if (next_release < loop_end) {
                next_release += period;
            }
        }
        std::this_thread::sleep_until(next_release);
    }
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    if (!bind_thread_cpu("ROBOT_INFERENCE_CPU", 4)) {
        rclcpp::shutdown();
        return;
    }
    struct sched_param sp{}; sp.sched_priority = 35;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for inference thread");
        rclcpp::shutdown();
        return;
    }
    const auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));
    auto next_release = std::chrono::steady_clock::now();

    while(rclcpp::ok()){
        next_release += period;
        auto loop_start = std::chrono::steady_clock::now();
        if(!is_running_.load()){
            const auto now = std::chrono::steady_clock::now();
            if (now > next_release) {
                const auto missed_periods = (now - next_release) / period;
                next_release += period * missed_periods;
                if (next_release < now) {
                    next_release += period;
                }
            }
            std::this_thread::sleep_until(next_release);
            continue;
        }

        try {
            std::unique_lock<std::mutex> mode_lock(mode_mutex_);
            if (!is_running_.load()) {
                mode_lock.unlock();
                const auto now = std::chrono::steady_clock::now();
                if (now > next_release) {
                    const auto missed_periods = (now - next_release) / period;
                    next_release += period * missed_periods;
                    if (next_release < now) {
                        next_release += period;
                    }
                }
                std::this_thread::sleep_until(next_release);
                continue;
            }
            auto& policy = active_policy();
            robot_->read_imu();
            update_obs_segments(policy.obs_segments, policy.obs_layout);
            publish_imu();
            publish_joint_states();
            if (supports_interrupt() && is_interrupt_.load() && !interrupt_topic_received_.load() &&
                joint_pos_buffer_.size() >= interrupt_action_.size()) {
                std::unique_lock<std::mutex> lock(interrupt_mutex_);
                const size_t offset = joint_pos_buffer_.size() - interrupt_action_.size();
                if (!interrupt_feedback_initialized_) {
                    std::copy(joint_pos_buffer_.begin() + offset, joint_pos_buffer_.end(),
                              interrupt_action_.begin());
                    interrupt_feedback_initialized_ = true;
                } else {
                    for (size_t i = 0; i < interrupt_action_.size(); ++i) {
                        const float feedback = joint_pos_buffer_[offset + i];
                        if (std::abs(feedback - interrupt_action_[i]) > interrupt_feedback_deadband_) {
                            interrupt_action_[i] = feedback;
                        }
                    }
                }
            }
            flatten_obs_segments(policy.obs_segments, policy.obs.begin());

            std::transform(policy.obs.begin(), policy.obs.end(), policy.obs.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });

            if (!policy.history_gather_plan.empty()) {
                if (policy.is_first_frame && policy.latent_loader) {
                    std::fill(policy.obs_history.begin(),
                              policy.obs_history.end() - policy.obs_num, 0.0f);
                    std::copy(policy.obs.begin(), policy.obs.end(),
                              policy.obs_history.end() - policy.obs_num);
                } else {
                    update_obs_history(policy.obs_history, policy.obs, policy.obs_num,
                                       policy.frame_stack, policy.is_first_frame);
                }
                gather_sparse_obs_history(policy.ctx->input_buffer, policy.obs_history,
                                          policy.history_gather_plan);
            } else {
                update_stacked_obs(policy.ctx->input_buffer, policy.obs, policy.obs_num,
                                   policy.frame_stack, policy.stack_order,
                                   policy.obs_layout_sizes, policy.is_first_frame);
            }
            if (policy.motion_loader) {
                step_motion_frame();
            }
            policy.is_first_frame = false;

            policy.ctx->session->Run(Ort::RunOptions{nullptr},
                policy.ctx->input_names_raw.data(), policy.ctx->input_tensor.get(), policy.ctx->num_inputs,
                policy.ctx->output_names_raw.data(), policy.ctx->output_tensor.get(), policy.ctx->num_outputs);

            {
                std::unique_lock<std::mutex> interrupt_lock(interrupt_mutex_, std::defer_lock);
                if (supports_interrupt() && is_interrupt_.load()) {
                    interrupt_lock.lock();
                }
                std::unique_lock<std::mutex> lock(act_mutex_);
                for (int i = 0; i < static_cast<int>(policy.ctx->output_buffer.size()); i++) {
                    policy.ctx->output_buffer[i] = action_rescale_ * std::clamp(
                        policy.ctx->output_buffer[i], -clip_actions_, clip_actions_);
                    const auto joint_idx = usd2urdf_[i];
                    act_[joint_idx] = policy.ctx->output_buffer[i] * action_scale_[joint_idx] +
                                      joint_default_angle_[joint_idx];
                }
                if (interrupt_lock.owns_lock()) {
                    for (size_t i = 0; i < interrupt_action_.size(); i++) {
                        act_[act_.size() - interrupt_action_.size() + i] = interrupt_action_[i];
                    }
                }
            }
            publish_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in inference thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        if (loop_end > next_release) {
            RCLCPP_WARN(this->get_logger(), "Inference loop overran! Took %lld us, but period is %lld us.", static_cast<long long>(elapsed_time.count()), static_cast<long long>(period.count()));
            const auto missed_periods = (loop_end - next_release) / period;
            next_release += period * missed_periods;
            if (next_release < loop_end) {
                next_release += period;
            }
        }
        std::this_thread::sleep_until(next_release);
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "mlockall failed.");
    }
    pthread_setname_np(pthread_self(), "main");
    std::shared_ptr<InferenceNode> node;
    try {
        node = std::make_shared<InferenceNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "Press 'X' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'A' to reset motors");
        RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
        RCLCPP_INFO(node->get_logger(), "Press 'Y' to switch between Gamepad Control / cmd_vel Control");
        if (node->supports_interrupt() || node->has_motion_policy() || node->has_latent_policy()){
            RCLCPP_INFO(node->get_logger(), "Press 'LB' to switch policy mode or start/reset latent playback");
        }
        if (node->has_motion_policy()){
            RCLCPP_INFO(node->get_logger(), "Press 'RB' to switch motion sequence (available in beyondmimic mode)");
        }
        RCLCPP_INFO(node->get_logger(), "Right Stick: Control forward, backward, left and right movement");
        RCLCPP_INFO(node->get_logger(), "LT/RT: Control turning (left / right rotation)");
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    node.reset();
    return 0;
}
