#pragma once

#include "close_chain_mapping.hpp"
#include <array>
#include <map>
#include <vector>

struct LinkParamsAtom01
{
    double l_rod;
    double l_bar;
    double l_spacing;
    Eigen::Vector3d r_A_0;
    Eigen::Vector3d r_B_0;
    Eigen::Vector3d r_C_0;
    double theta_0;
};

struct IKResultAtom01
{
    std::array<Eigen::Vector3d, 2> r_bar;
    std::array<Eigen::Vector3d, 2> r_rod;
    std::array<Eigen::Vector3d, 2> r_C;
    Eigen::Vector2d THETA;
};

class DecoupleAtom01 : public Decouple
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DecoupleAtom01();

    void get_forwardQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel,
                        Eigen::VectorXd &tau, bool is_left) override;

    void get_decoupleQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel,
                         Eigen::VectorXd &tau, bool is_left) override;

private:
    std::vector<LinkParamsAtom01> links_left_;
    std::vector<LinkParamsAtom01> links_right_;
    std::map<bool, Eigen::Vector2d> last_solution_;

    static std::vector<LinkParamsAtom01> get_links(bool is_left);

    const std::vector<LinkParamsAtom01> &cached_links(bool is_left) const
    {
        return is_left ? links_left_ : links_right_;
    }

    IKResultAtom01 inverse_kinematics(double q_roll, double q_pitch, bool is_left);
    JacobianResult jacobian(const IKResultAtom01 &ik_result, double q_pitch);
    std::pair<Eigen::Vector2d, JacobianResult> get_decouple(double roll, double pitch, bool is_left);
    ForwardMappingResult forward_kinematics(const Eigen::Vector2d &thetaRef, bool is_left);
};
