#ifndef _MANIP_FILTER_
#define _MANIP_FILTER_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <lbr_fri_idl/msg/lbr_joint_position_command.hpp>


#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <Eigen/Dense>

#include <fstream>
#include <string>

class RobotDynamics
{
    private:

    // Robot state
    Eigen::VectorXd q_init_, q_model_, q_dot_model_;
    Eigen::VectorXd q_desired_;
    Eigen::VectorXd q_next_;
    Eigen::MatrixXd mass_matrix_;
    Eigen::VectorXd gravity_;
    Eigen::VectorXd coriolis_;

    // Robot params
    size_t dof_;

    // Control params
    Eigen::MatrixXd Kp_ ;
    Eigen::MatrixXd Kd_;

    // KDL-related members
    KDL::Tree kdl_tree_;
    KDL::Chain kdl_chain_;
    std::shared_ptr<KDL::ChainDynParam> dyn_param_solver_;

    void loadURDF(const std::string &urdf_path);
    void buildKDLChain(const std::string &base_link, const std::string &tip_link);
    void initializeSolvers();

    Eigen::VectorXd computeTorques(const Eigen::VectorXd &q_desired, const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot);
    Eigen::VectorXd computeDirectDynamics(const Eigen::VectorXd &torques);

    Eigen::VectorXd stdVectorToEigen(const std::vector<double>& vec) {
        return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
    }

    std::vector<double> eigenToStdVector(const Eigen::VectorXd& vec) {
        return std::vector<double>(vec.data(), vec.data() + vec.size());
    }

    bool q_init_set = false;
    bool gains_set = false;
    bool first_response = false;

public:
    RobotDynamics();
    bool set_q_init(const std::vector<double>);
    bool set_gains(const std::vector<double>, const std::vector<double>);
    std::vector<double> manipulator_response(std::vector<double> q_desired, std::vector<double> q_actual,std::vector<double> q_dot_actual, double dt);

};

#endif