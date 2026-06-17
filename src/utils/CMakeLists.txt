# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

AUX_SOURCE_DIRECTORY(. SOURCE_LIST_UTILS)
add_library(utils
        STATIC
        ${SOURCE_LIST_UTILS}
)
target_include_directories(utils PUBLIC ./ ${CNPY_INCLUDE_DIR})
target_link_libraries(utils PUBLIC ${PUBLIC_DEPENDENCIES})