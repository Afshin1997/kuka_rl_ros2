#include "catch_and_throw/manip_filter.h"

RobotDynamics::RobotDynamics(){
    try{
        loadURDF("/home/user/kuka_rl_ros2/src/catch_and_throw/lbr-description-isaac-sim/lbr_iiwa7.urdf");
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        exit(1);
    }
    buildKDLChain("iiwa_link_0", "planar_end_eff_link");
    initializeSolvers(); 
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
    q_model_ = Eigen::VectorXd::Zero(dof_);
    q_dot_model_ = Eigen::VectorXd::Zero(dof_);
    q_desired_ = Eigen::VectorXd::Zero(dof_);
    q_next_ = Eigen::VectorXd::Zero(dof_);
    mass_matrix_ = Eigen::MatrixXd::Zero(dof_, dof_);
    gravity_ = Eigen::VectorXd::Zero(dof_);
    coriolis_ = Eigen::VectorXd::Zero(dof_);
    Kp_ = Eigen::MatrixXd::Zero(dof_, dof_);
    Kd_ = Eigen::MatrixXd::Zero(dof_, dof_);
}

void RobotDynamics::initializeSolvers(){
    KDL::Vector gravity(0.0, 0.0, -9.81);
    dyn_param_solver_ = std::make_shared<KDL::ChainDynParam>(kdl_chain_, gravity);
}


Eigen::VectorXd RobotDynamics::computeTorques(const Eigen::VectorXd &q_desired, const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot){
    KDL::JntArray q_kdl(dof_), q_dot_kdl(dof_);
    for (size_t i = 0; i < dof_; ++i)
    {
        q_kdl(i) = q_model_(i);
        q_dot_kdl(i) = q_dot_model_(i);
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

    
    return M*(Kp_ * (q_desired - q) - Kd_*q_dot) + G + C;
}

Eigen::VectorXd RobotDynamics::computeDirectDynamics(const Eigen::VectorXd &torques){
    Eigen::VectorXd q_ddot;
    KDL::JntArray q_kdl(dof_), q_dot_kdl(dof_);
    for (size_t i = 0; i < dof_; ++i)
    {
        q_kdl(i) = q_model_(i);
        q_dot_kdl(i) = q_dot_model_(i);
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

bool RobotDynamics::set_q_init(const std::vector<double> q_init){
    if(q_init.size() != dof_){
        return false;
    }
    else{
        q_init_ = stdVectorToEigen(q_init); 
        q_init_set = true;
        return true;
    }
}
bool RobotDynamics::set_gains(const std::vector<double> kp, const std::vector<double> kd){
    if(kp.size()!=dof_ || kd.size()!=dof_){
        return false;
    }
    for (int i=0; i<dof_; i++){
        Kp_(i,i) = kp[i];
        Kd_(i,i) = kd[i];
    }   
    gains_set = true;
    return true;
}

std::vector<double> RobotDynamics::manipulator_response(std::vector<double> q_desired, std::vector<double> q_actual,std::vector<double> q_dot_actual, double dt){
    std::vector<double> q_response;

    if(!first_response){
        q_model_ = q_init_;
        first_response = true;
    }
    
    if (q_init_set && gains_set && q_desired.size() == dof_ && q_actual.size() == dof_){
        //Considering the REAL manipulator state and the desired one, 
        //we compute the torques as done in simulation with a PD law
        Eigen::VectorXd torques = computeTorques(stdVectorToEigen(q_desired), stdVectorToEigen(q_actual), stdVectorToEigen(q_dot_actual));
        //Considering then the dynamic MODEL of the manipulator, we compute its response
        //to that torques, obtaining acceleration, velocities, position
        Eigen::VectorXd q_ddot_model = computeDirectDynamics(torques);
        q_dot_model_ = q_dot_model_ + q_ddot_model * dt;
        q_model_ = q_model_ + q_dot_model_ * dt;
        q_response = eigenToStdVector(q_model_);
    }

    return q_response;

}
