#pragma once

#include "close_chain_mapping.hpp"
#include <array>
#include <map>
#include <vector>

struct LinkParamsRP1
{
    double l_rod;
    double l_bar;
    Eigen::Vector3d p_o_a;
    Eigen::Vector3d p_o_b;
    Eigen::Vector3d p_o_c;
    double theta_0;
};

struct IKResultRP1
{
    std::array<Eigen::Vector3d, 2> r_bar;
    std::array<Eigen::Vector3d, 2> r_rod;
    std::array<Eigen::Vector3d, 2> db_dp;
    std::array<Eigen::Vector3d, 2> db_dr;
    Eigen::Vector2d THETA;
};

class DecoupleRP1 : public Decouple
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DecoupleRP1();

    void get_forwardQVT(Eigen::Vector2d &q, Eigen::Vector2d &vel,
                        Eigen::Vector2d &tau, bool is_left) override;

    void get_decoupleQVT(Eigen::Vector2d &q, Eigen::Vector2d &vel,
                         Eigen::Vector2d &tau, bool is_left) override;

private:
    std::vector<LinkParamsRP1> links_left_;
    std::vector<LinkParamsRP1> links_right_;
    std::map<bool, Eigen::Vector2d> last_solution_;

    static std::vector<LinkParamsRP1> get_links(bool is_left);

    const std::vector<LinkParamsRP1> &cached_links(bool is_left) const
    {
        return is_left ? links_left_ : links_right_;
    }

    IKResultRP1 inverse_kinematics(double q_pitch, double q_roll, bool is_left);
    JacobianResult jacobian(const IKResultRP1 &ik_result);
    std::pair<Eigen::Vector2d, JacobianResult> get_decouple(double pitch, double roll, bool is_left);
    ForwardMappingResult forward_kinematics(const Eigen::Vector2d &thetaRef, bool is_left);
};
