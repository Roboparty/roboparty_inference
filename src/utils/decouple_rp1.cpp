#include "decouple_rp1.hpp"
#include <iostream>

static const Eigen::Vector3d S_BAR(-1.0, 0.0, 0.0);

//////********************link parameters********************//////
std::vector<LinkParamsRP1> DecoupleRP1::get_links(bool is_left)
{
    std::vector<LinkParamsRP1> links(2);

    if (is_left)
    {
        // long link
        links[0].l_rod = 210.636;
        links[0].l_bar = 26.0;
        links[0].p_o_a = Eigen::Vector3d(0.0, 0.0, -19.0);
        links[0].p_o_b = Eigen::Vector3d(-27.5, -15.0, -0.6);
        links[0].p_o_c = Eigen::Vector3d(-37.390, 0, 209.663);
        links[0].theta_0 = -180.0 * M_PI / 180.0;

        // short link
        links[1].l_rod = 139.941;
        links[1].l_bar = 26.0;
        links[1].p_o_a = Eigen::Vector3d(0.0, 0.0, -19.0);
        links[1].p_o_b = Eigen::Vector3d(-27.5, 15.0, -0.6);
        links[1].p_o_c = Eigen::Vector3d(-37.390, 0, 138.663);
        links[1].theta_0 = 0.0;
    }
    else
    {
        // long link
        links[0].l_rod = 210.636;
        links[0].l_bar = 26.0;
        links[0].p_o_a = Eigen::Vector3d(0.0, 0.0, -19.0);
        links[0].p_o_b = Eigen::Vector3d(-27.5, 15.0, -0.6);
        links[0].p_o_c = Eigen::Vector3d(-37.390, 0, 209.663);
        links[0].theta_0 = 0.0;

        // short link
        links[1].l_rod = 139.941;
        links[1].l_bar = 26.0;
        links[1].p_o_a = Eigen::Vector3d(0.0, 0.0, -19.0);
        links[1].p_o_b = Eigen::Vector3d(-27.5, -15.0, -0.6);
        links[1].p_o_c = Eigen::Vector3d(-37.390, 0, 138.663);
        links[1].theta_0 = -180.0 * M_PI / 180.0;
    }

    return links;
}

DecoupleRP1::DecoupleRP1()
    : links_left_(get_links(true)), links_right_(get_links(false))
{
}
//////********************link parameters********************//////

//////********************inverse kinematics*****************//////
IKResultRP1
DecoupleRP1::inverse_kinematics(double q_pitch, double q_roll, bool is_left)
{
    IKResultRP1 result;
    result.THETA = Eigen::Vector2d::Zero();

    double sp = std::sin(q_pitch), cp = std::cos(q_pitch);
    double sr = std::sin(q_roll), cr = std::cos(q_roll);

    // R_y (pitch)
    Eigen::Matrix3d R_y;
    R_y << cp, 0, sp,
        0, 1, 0,
        -sp, 0, cp;

    // R_x (roll)
    Eigen::Matrix3d R_x;
    R_x << 1, 0, 0,
        0, cr, -sr,
        0, sr, cr;

    // 旋转矩阵导数
    Eigen::Matrix3d dRy_dp;
    dRy_dp << -sp, 0, cp,
        0, 0, 0,
        -cp, 0, -sp;

    Eigen::Matrix3d dRx_dr;
    dRx_dr << 0, 0, 0,
        0, -sr, -cr,
        0, cr, -sr;

    const auto &links = cached_links(is_left);

    for (int i = 0; i < 2; i++)
    {
        double l_rod = links[i].l_rod;
        double l_bar = links[i].l_bar;
        Eigen::Vector3d p_a = links[i].p_o_a;
        Eigen::Vector3d p_b = links[i].p_o_b;
        Eigen::Vector3d p_c = links[i].p_o_c;
        double theta_0 = links[i].theta_0;

        // d = p_b - p_a
        Eigen::Vector3d d = p_b - p_a;

        // b_rot = R_y @ p_a + R_y @ R_x @ d
        Eigen::Vector3d b_rot = R_y * p_a + R_y * R_x * d;

        // diff = p_c - b_rot
        Eigen::Vector3d diff = p_c - b_rot;

        // 约束方程 A·cosθ + B·sinθ + C = 0
        double A = 2.0 * l_bar * diff[1];
        double B = 2.0 * S_BAR[0] * l_bar * diff[2];
        double C = diff.dot(diff) + l_bar * l_bar - l_rod * l_rod;

        // 求解 theta
        double R_val = std::sqrt(A * A + B * B);
        double phi = std::atan2(B, A);
        double cos_arg = -C / R_val;

        if (std::abs(cos_arg) > 1.0)
        {
            std::cerr << "Warning: 无解: |-C/R| = " << std::abs(cos_arg) << " > 1" << std::endl;
            cos_arg = std::max(-1.0, std::min(1.0, cos_arg));
        }

        double delta = std::acos(cos_arg);
        double t1 = phi + delta - theta_0;
        double t2 = phi - delta - theta_0;

        double theta_val = std::abs(t1) < std::abs(t2) ? t1 : t2;

        // 中间几何量
        double theta_actual = theta_val + theta_0;
        double ct = std::cos(theta_actual);
        double st = std::sin(theta_actual);

        Eigen::Vector3d r_bar_i(0.0, l_bar * ct, S_BAR[0] * l_bar * st);
        Eigen::Vector3d c_rot = r_bar_i + p_c;
        Eigen::Vector3d r_rod_i = b_rot - c_rot;

        // ∂b_rot/∂pitch, ∂b_rot/∂roll
        Eigen::Vector3d db_dp_i = dRy_dp * (p_a + R_x * d);
        Eigen::Vector3d db_dr_i = R_y * dRx_dr * d;

        result.r_bar[i] = r_bar_i;
        result.r_rod[i] = r_rod_i;
        result.db_dp[i] = db_dp_i;
        result.db_dr[i] = db_dr_i;
        result.THETA[i] = theta_val;
    }

    return result;
}
//////********************inverse kinematics*****************//////

//////********************jacobian***************************//////
JacobianResult
DecoupleRP1::jacobian(const IKResultRP1 &ik_result)
{
    const auto &r_bar = ik_result.r_bar;
    const auto &r_rod = ik_result.r_rod;
    const auto &db_dp = ik_result.db_dp;
    const auto &db_dr = ik_result.db_dr;

    Eigen::Matrix2d J_theta;
    J_theta << r_rod[0].dot(S_BAR.cross(r_bar[0])), 0,
        0, r_rod[1].dot(S_BAR.cross(r_bar[1]));

    Eigen::Matrix2d J_Temp;
    J_Temp << r_rod[0].dot(db_dp[0]), r_rod[0].dot(db_dr[0]),
        r_rod[1].dot(db_dp[1]), r_rod[1].dot(db_dr[1]);

    JacobianResult result;
    result.J_motor2Joint = J_Temp.inverse() * J_theta;
    result.J_Joint2motor = J_theta.inverse() * J_Temp;

    return result;
}
//////********************jacobian***************************//////

// from joint [pitch, roll] to motor [theta], from S to P
std::pair<Eigen::Vector2d, JacobianResult>
DecoupleRP1::get_decouple(double pitch, double roll, bool is_left)
{
    IKResultRP1 kinematics = inverse_kinematics(pitch, roll, is_left);
    JacobianResult Jac = jacobian(kinematics);
    return {kinematics.THETA, Jac};
}

//////********************forward kinematics*****************//////
ForwardMappingResult
DecoupleRP1::forward_kinematics(const Eigen::Vector2d &thetaRef, bool is_left)
{
    ForwardMappingResult mapping_result;

    int count = 0;
    Eigen::Vector2d f_error{10, 10};
    Eigen::Vector2d x_k = last_solution_.count(is_left) ? last_solution_[is_left] : Eigen::Vector2d::Zero();

    JacobianResult Jac;
    static constexpr int MAX_ITERATIONS = 100;
    static constexpr double TOLERANCE = 1e-3;
    static constexpr double ALPHA = 0.5;

    while (f_error.norm() > TOLERANCE && count < MAX_ITERATIONS)
    {
        double cur_pitch = x_k[0];
        double cur_roll = x_k[1];

        IKResultRP1 kinematics = inverse_kinematics(cur_pitch, cur_roll, is_left);
        Jac = jacobian(kinematics);

        if (Jac.J_motor2Joint.hasNaN())
        {
            std::cerr << "DecoupleRP1::forward_kinematics() Jac is nan!!" << std::endl;
            std::cerr << "  pitch=" << cur_pitch << ", roll=" << cur_roll << std::endl;
            mapping_result.ankle_joint_ori = Eigen::Vector2d::Zero();
            mapping_result.Jac = Jac;
            return mapping_result;
        }

        f_error = thetaRef - kinematics.THETA;
        x_k = x_k + ALPHA * Jac.J_motor2Joint * f_error;
        count++;
    }

    if (f_error.norm() < TOLERANCE)
    {
        last_solution_[is_left] = x_k;
    }

    mapping_result.ankle_joint_ori = x_k;
    mapping_result.Jac = Jac;

    return mapping_result;
}
//////********************forward kinematics*****************//////

// from joint [pitch, roll] to motor [theta], from Serial to Parallel
void DecoupleRP1::get_decoupleQVT(Eigen::Vector2d &q, Eigen::Vector2d &vel, Eigen::Vector2d &tau, bool is_left)
{
    double pitch = q[0]; // rotation axis [0 1 0]
    double roll = q[1];

    auto motor = get_decouple(pitch, roll, is_left);
    q = motor.first;
    vel = motor.second.J_Joint2motor * vel;
    tau = motor.second.J_motor2Joint.transpose() * tau;
}

void DecoupleRP1::get_forwardQVT(Eigen::Vector2d &q, Eigen::Vector2d &vel, Eigen::Vector2d &tau, bool is_left)
{
    ForwardMappingResult joint = forward_kinematics(q, is_left);
    q = joint.ankle_joint_ori;
    vel = joint.Jac.J_motor2Joint * vel;
    tau = joint.Jac.J_Joint2motor.transpose() * tau;
}
