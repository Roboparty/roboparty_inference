// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "robot_interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(robot_py, m) {
    py::class_<RobotInterface>(m, "RobotInterface")
        .def(py::init<const std::string&>(), py::arg("config_file"))
        .def("apply_action", &RobotInterface::apply_action,
             py::arg("p"),
             py::arg("v")   = std::vector<float>{},
             py::arg("kp")  = std::vector<float>{},
             py::arg("kd")  = std::vector<float>{},
             py::arg("tau") = std::vector<float>{})
        .def("init_motors", &RobotInterface::init_motors)
        .def("deinit_motors", &RobotInterface::deinit_motors)
        .def("reset_joints", &RobotInterface::reset_joints, py::arg("joint_default_angle"))
        .def("set_zeros", &RobotInterface::set_zeros)
        .def("clear_errors", &RobotInterface::clear_errors)
        .def("refresh_joints", &RobotInterface::refresh_joints)
        .def("get_joint_q", &RobotInterface::get_joint_q)
        .def("get_joint_vel", &RobotInterface::get_joint_vel)
        .def("get_joint_tau", &RobotInterface::get_joint_tau)
        .def("get_quat", &RobotInterface::get_quat)
        .def("get_ang_vel", &RobotInterface::get_ang_vel)
        .def_property_readonly("is_init", [](const RobotInterface &r) {
            return r.is_init_.load();
        });
}
