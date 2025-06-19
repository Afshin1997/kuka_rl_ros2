#ifndef OPTITRACK_LISTENER_CPP_FILTERS_HPP
#define OPTITRACK_LISTENER_CPP_FILTERS_HPP

#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <cmath>

namespace optitrack_listener_cpp {

// Moving average filter for position smoothing
class MovingAverageFilter {
private:
    std::deque<Eigen::Vector3d> buffer_;
    size_t window_size_;
    bool initialized_;
    
public:
    explicit MovingAverageFilter(size_t window_size = 5);
    
    Eigen::Vector3d filter(const Eigen::Vector3d& input);
    void reset();
    bool is_initialized() const { return initialized_; }
};

// Savitzky-Golay filter for smooth derivatives
class SavitzkyGolayFilter {
private:
    size_t window_length_;
    int poly_order_;
    int deriv_order_;
    double delta_;
    
    std::deque<Eigen::Vector3d> buffer_;
    Eigen::VectorXd coefficients_;
    bool initialized_;
    bool buffer_full_;
    
    void compute_coefficients();
    
public:
    SavitzkyGolayFilter(size_t window_length = 7, int poly_order = 2, 
                       int deriv_order = 0, double delta = 1.0);
    
    Eigen::Vector3d filter(const Eigen::Vector3d& input);
    void reset();
    bool is_initialized() const { return initialized_; }
    bool is_ready() const { return buffer_full_; }
};

// Kalman filter for more advanced filtering
class KalmanFilter {
private:
    // State: [x, y, z, vx, vy, vz]
    Eigen::VectorXd x_;  // State vector
    Eigen::MatrixXd P_;  // Error covariance
    Eigen::MatrixXd F_;  // State transition
    Eigen::MatrixXd H_;  // Measurement matrix
    Eigen::MatrixXd Q_;  // Process noise
    Eigen::MatrixXd R_;  // Measurement noise
    
    double dt_;
    bool initialized_;
    
public:
    explicit KalmanFilter(double dt = 0.005);
    
    void predict();
    void update(const Eigen::Vector3d& measurement);
    
    Eigen::Vector3d get_position() const;
    Eigen::Vector3d get_velocity() const;
    
    void reset();
    bool is_initialized() const { return initialized_; }
    
    void set_process_noise(double position_noise, double velocity_noise);
    void set_measurement_noise(double noise);
};

// Velocity estimator using finite differences
class VelocityEstimator {
private:
    double dt_;
    Eigen::Vector3d last_position_;
    Eigen::Vector3d velocity_;
    bool initialized_;
    
    // For smoothing velocity estimates
    MovingAverageFilter velocity_filter_;
    
public:
    explicit VelocityEstimator(double dt = 0.005, size_t filter_window = 3);
    
    Eigen::Vector3d update(const Eigen::Vector3d& position);
    void reset();
    
    Eigen::Vector3d get_velocity() const { return velocity_; }
    bool is_initialized() const { return initialized_; }
};

// Outlier rejection filter
class OutlierFilter {
private:
    double max_jump_distance_;
    double max_velocity_;
    Eigen::Vector3d last_valid_position_;
    bool initialized_;
    
public:
    OutlierFilter(double max_jump = 0.1, double max_vel = 10.0);
    
    bool is_valid(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity);
    void update_last_valid(const Eigen::Vector3d& position);
    void reset();
};

} // namespace optitrack_listener_cpp

#endif // OPTITRACK_LISTENER_CPP_FILTERS_HPP