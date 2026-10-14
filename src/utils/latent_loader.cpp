// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "latent_loader.hpp"
#include "npz_loader.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

LatentLoader::LatentLoader(const std::string& latent_file, float gamma, int window_size)
    : latent_(roboparty::load_npz(latent_file, "z")), gamma_(gamma) {
    if (latent_.shape.size() != 2 || latent_.shape[0] == 0 || latent_.shape[1] == 0 ||
        latent_.word_size != sizeof(float)) {
        throw std::runtime_error("latent z must be a non-empty float32 [frames, dim] array");
    }
    if (window_size <= 0) {
        throw std::runtime_error("window_size must be positive");
    }
    window_size_ = static_cast<size_t>(window_size);
}

void LatentLoader::next(std::vector<float>& output) {
    if (output.size() != get_dim()) {
        throw std::runtime_error("latent observation size does not match latent dimension");
    }

    std::fill(output.begin(), output.end(), 0.0f);
    const size_t count = std::min(window_size_, get_num_frames() - frame_);
    float weight = 1.0f;
    float weight_sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float* z = latent_.data<float>() + (frame_ + i) * get_dim();
        weight_sum += weight;
        for (size_t j = 0; j < output.size(); ++j) {
            output[j] += weight * z[j];
        }
        weight *= gamma_;
    }

    const float* first_z = latent_.data<float>();
    float first_norm_sq = 0.0f;
    float output_norm_sq = 0.0f;
    for (size_t j = 0; j < output.size(); ++j) {
        output[j] /= weight_sum;
        first_norm_sq += first_z[j] * first_z[j];
        output_norm_sq += output[j] * output[j];
    }
    if (first_norm_sq > 0.0f && output_norm_sq > 0.0f) {
        const float scale = std::sqrt(first_norm_sq / output_norm_sq);
        for (float& value : output) {
            value *= scale;
        }
    }

    if (playing_ && ++frame_ == get_num_frames()) {
        frame_ = get_num_frames() - 1;
        playing_ = false;
    }
}
