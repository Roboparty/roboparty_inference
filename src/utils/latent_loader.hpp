// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <cnpy.h>

#include <cstddef>
#include <string>
#include <vector>

class LatentLoader {
public:
    LatentLoader(const std::string& latent_file, float gamma, int window_size);

    size_t get_num_frames() const { return latent_.shape[0]; }
    size_t get_dim() const { return latent_.shape[1]; }
    void next(std::vector<float>& output);
    // Reset to the first frame and wait for the playback trigger.
    void reset() { frame_ = 0; playing_ = false; }
    // LB behavior: start from frame zero on the initial stop; rewind and stop
    // when playing or after reaching the terminal frame.
    void trigger() {
        if (playing_ || frame_ != 0) {
            frame_ = 0;
            playing_ = false;
        } else {
            playing_ = true;
        }
    }
    bool is_playing() const { return playing_; }

private:
    cnpy::NpyArray latent_;
    float gamma_;
    size_t window_size_;
    size_t frame_ = 0;
    bool playing_ = false;
};
