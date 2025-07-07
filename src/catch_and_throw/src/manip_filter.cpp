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
    Eigen::VectorXd q_init_, q_actual_, q_dot_actual_;
    Eigen::VectorXd q_desired_;
    Eigen::VectorXd q_next_;
    Eigen::MatrixXd mass_matrix_;
    Eigen::VectorXd gravity_;
    Eigen::VectorXd coriolis_;

    // Robot params
    size_t dof_;

    // Control params
    double Kp_ = 100.0;
    double Kd_ = 20.0;

    // KDL-related members
    KDL::Tree kdl_tree_;
    KDL::Chain kdl_chain_;
    std::shared_ptr<KDL::ChainDynParam> dyn_param_solver_;

    void loadURDF(const std::string &urdf_path);
    void buildKDLChain(const std::string &base_link, const std::string &tip_link);
    void initializeSolvers();

    Eigen::VectorXd computeTorques(const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot);
    Eigen::VectorXd computeDirectDynamics(const Eigen::VectorXd &torques);

    Eigen::VectorXd stdVectorToEigen(const std::vector<double>& vec) {
        return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
    }

    std::vector<double> eigenToStdVector(const Eigen::VectorXd& vec) {
        return std::vector<double>(vec.data(), vec.data() + vec.size());
    }

public:
    RobotDynamics(const std::string &urdf_path, const std::string &base_link, const std::string &tip_link, 
        double kp, double kd, std::vector<double> q_init);
    std::vector<double> manipulator_response(std::vector<double> q_cmd, std::vector<double> q_actual, double dt);

};

RobotDynamics::RobotDynamics(const std::string &urdf_path, const std::string &base_link, 
    const std::string &tip_link, double kp, double kd, std::vector<double> q_init){
    loadURDF(urdf_path);
    buildKDLChain(base_link, tip_link);
    initializeSolvers();
    Kp_ = kp;
    Kd_ = kd;
    if(q_init.size() != dof_){
        exit(1);
    }
    else
        q_init_ = stdVectorToEigen(q_init);     
   
}


void RobotDynamics::loadURDF(const std::string &urdf_path){
    if (!kdl_parser::treeFromFile(urdf_path, kdl_tree_))
        throw std::runtime_error("Failed to parse URDF file using KDL: " + urdf_path);
}

void RobotDynamics::buildKDLChain(const std::string &base_link, const std::string &tip_link){
    if (!kdl_tree_.getChain(base_link, tip_link, kdl_chain_))
        throw std::runtime_error("Failed to extract KDL chain");

    dof_ = kdl_chain_.getNrOfJoints();

    // Initialize state vectors
    q_init_ = Eigen::VectorXd::Zero(dof_);
    q_actual_ = Eigen::VectorXd::Zero(dof_);
    q_dot_actual_ = Eigen::VectorXd::Zero(dof_);
    q_desired_ = Eigen::VectorXd::Zero(dof_);
    q_next_ = Eigen::VectorXd::Zero(dof_);
    mass_matrix_ = Eigen::MatrixXd::Zero(dof_, dof_);
    gravity_ = Eigen::VectorXd::Zero(dof_);
    coriolis_ = Eigen::VectorXd::Zero(dof_);
}

void RobotDynamics::initializeSolvers(){
    KDL::Vector gravity(0.0, 0.0, -9.81);
    dyn_param_solver_ = std::make_shared<KDL::ChainDynParam>(kdl_chain_, gravity);
}


Eigen::VectorXd RobotDynamics::computeTorques(const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot){
    return Kp_ * (q_desired_ - q) - Kd_ * q_dot;
}

Eigen::VectorXd RobotDynamics::computeDirectDynamics(const Eigen::VectorXd &torques){
    Eigen::VectorXd q_ddot;
    KDL::JntArray q_kdl(dof_), q_dot_kdl(dof_);
    for (size_t i = 0; i < dof_; ++i)
    {
        q_kdl(i) = q_actual_(i);
        q_dot_kdl(i) = q_dot_actual_(i);
    }

    KDL::JntSpaceInertiaMatrix M_kdl(dof_);
    KDL::JntArray C_kdl(dof_), G_kdl(dof_);
    dyn_param_solver_->JntToMass(q_kdl, M_kdl);
    dyn_param_solver_->JntToCoriolis(q_kdl, q_dot_kdl, C_kdl);
    dyn_param_solver_->JntToGravity(q_kdl, G_kdl);

    Eigen::MatrixXd M(dof_, dof_);
    Eigen::VectorXd C(dof_), G(dof_);
    for (size_t i = 0; i < dof_; ++i)
    {
        C(i) = C_kdl(i);
        G(i) = G_kdl(i);
        for (size_t j = 0; j < dof_; ++j)
            M(i, j) = M_kdl(i, j);
    }

    q_ddot = M.inverse() * (torques - C - G);
    return q_ddot;
}

std::vector<double> manipulator_response(std::vector<double> q_cmd, std::vector<double> q_actual, double dt){

}
