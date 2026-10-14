// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "close_chain_mapping.hpp"
#include "decouple_rp1.3.hpp"
#include "decouple_rpo.hpp"

std::shared_ptr<Decouple> Decouple::create(const std::string &type)
{
    if (type == "rp1.3")
    {
        return std::make_shared<DecoupleRP1_3>();
    }
    else if (type == "rpo")
    {
        return std::make_shared<DecoupleRPO>();
    }
    else
    {
        throw std::runtime_error("Unknown close_chain type: " + type +
                                 ". Supported types: rp1.3, rpo");
    }
}
