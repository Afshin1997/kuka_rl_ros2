#include "optitrack_listener_cpp/filters.hpp"

namespace optitrack_listener_cpp {

// MovingAverageFilter implementation
MovingAverageFilter::MovingAverageFilter(size_t window_size)
    : window_size_(window_size), initialized_(false) {
    if (window_size_ == 0) {
        window_size_ = 1;
    }
}

Eigen::Vector3d MovingAverageFilter::filter(const Eigen::Vector3d& input) {
    if (!initialized_) {
        buffer_.assign(window_size_, input);
        initialized_ = true;
    }
    
    // Remove oldest value and add new one
    buffer_.pop_front();
    buffer_.push_back(input);
    
    // Calculate average
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& val : buffer_) {
        sum += val;
    }
    
    return sum / static_cast<double>(buffer_.size());
}

void MovingAverageFilter::reset() {
    buffer_.clear();
    initialized_ = false;
}

// SavitzkyGolayFilter implementation
SavitzkyGolayFilter::SavitzkyGolayFilter(size_t window_length, int poly_order, 
                                       int deriv_order, double delta)
    : window_length_(window_length), poly_order_(poly_order), 
      deriv_order_(deriv_order), delta_(delta), 
      initialized_(false), buffer_full_(false) {
    
    // Ensure window_length is odd
    if (window_length_ % 2 == 0) {
        window_length_ += 1;
    }
    
    // Ensure poly_order is valid
    if (poly_order_ >= static_cast<int>(window_length_)) {
        poly_order_ = window_length_ - 1;
    }
    
    compute_coefficients();
}

void SavitzkyGolayFilter::compute_coefficients() {
    int m = (window_length_ - 1) / 2;
    
    // Build the design matrix A
    Eigen::MatrixXd A(window_length_, poly_order_ + 1);
    for (int i = 0; i < static_cast<int>(window_length_); ++i) {
        double x = (i - m) * delta_;
        double xi = 1.0;
        for (int j = 0; j <= poly_order_; ++j) {
            A(i, j) = xi;
            xi *= x;
        }
    }
    
    // Compute the pseudo-inverse and extract the row for the derivative
    Eigen::MatrixXd ATA = A.transpose() * A;
    Eigen::MatrixXd ATAinv = ATA.inverse();
    Eigen::MatrixXd C = ATAinv * A.transpose();
    
    // For real-time filtering, we want coefficients for the last point
    coefficients_ = C.row(deriv_order_).transpose();
    
    // Apply factorial scaling for derivatives
    if (deriv_order_ > 0) {
        double factorial = 1.0;
        for (int i = 1; i <= deriv_order_; ++i) {
            factorial *= i;
        }
        coefficients_ *= factorial / std::pow(delta_, deriv_order_);
    }
}

Eigen::Vector3d SavitzkyGolayFilter::filter(const Eigen::Vector3d& input) {
    // Add to buffer
    buffer_.push_back(input);
    
    // Keep buffer size limited
    if (buffer_.size() > window_length_) {
        buffer_.pop_front();
        buffer_full_ = true;
    }
    
    if (!initialized_) {
        initialized_ = true;
    }
    
    // If buffer not full yet, return input (or could pad with first value)
    if (!buffer_full_) {
        return input;
    }
    
    // Apply Savitzky-Golay filter
    Eigen::Vector3d filtered_value = Eigen::Vector3d::Zero();
    
    for (size_t i = 0; i < buffer_.size(); ++i) {
        filtered_value += buffer_[i] * coefficients_(i);
    }
    
    return filtered_value;
}

void SavitzkyGolayFilter::reset() {
    buffer_.clear();
    initialized_ = false;
    buffer_full_ = false;
}

// KalmanFilter implementation
KalmanFilter::KalmanFilter(double dt) : dt_(dt), initialized_(false) {
    // State vector: [x, y, z, vx, vy, vz]
    x_ = Eigen::VectorXd::Zero(6);
    
    // State transition matrix
    F_ = Eigen::MatrixXd::Identity(6, 6);
    F_(0, 3) = dt_;
    F_(1, 4) = dt_;
    F_(2, 5) = dt_;
    
    // Measurement matrix (we only measure position)
    H_ = Eigen::MatrixXd::Zero(3, 6);
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;
    
    // Initial covariance
    P_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;
    
    // Process noise
    set_process_noise(0.01, 0.1);
    
    // Measurement noise
    set_measurement_noise(0.01);
}

void KalmanFilter::predict() {
    // Predict state
    x_ = F_ * x_;
    
    // Predict covariance
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::update(const Eigen::Vector3d& measurement) {
    if (!initialized_) {
        x_.head<3>() = measurement;
        x_.tail<3>() = Eigen::Vector3d::Zero();
        initialized_ = true;
        return;
    }
    
    // Innovation
    Eigen::Vector3d y = measurement - H_ * x_;
    
    // Innovation covariance
    Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;
    
    // Kalman gain
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    
    // Update state
    x_ = x_ + K * y;
    
    // Update covariance
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
}

Eigen::Vector3d KalmanFilter::get_position() const {
    return x_.head<3>();
}

Eigen::Vector3d KalmanFilter::get_velocity() const {
    return x_.tail<3>();
}

void KalmanFilter::reset() {
    x_ = Eigen::VectorXd::Zero(6);
    P_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;
    initialized_ = false;
}

void KalmanFilter::set_process_noise(double position_noise, double velocity_noise) {
    Q_ = Eigen::MatrixXd::Zero(6, 6);
    
    // Position noise
    Q_(0, 0) = position_noise * dt_ * dt_;
    Q_(1, 1) = position_noise * dt_ * dt_;
    Q_(2, 2) = position_noise * dt_ * dt_;
    
    // Velocity noise
    Q_(3, 3) = velocity_noise * dt_;
    Q_(4, 4) = velocity_noise * dt_;
    Q_(5, 5) = velocity_noise * dt_;
}

void KalmanFilter::set_measurement_noise(double noise) {
    R_ = Eigen::Matrix3d::Identity() * noise;
}

// VelocityEstimator implementation
VelocityEstimator::VelocityEstimator(double dt, size_t filter_window)
    : dt_(dt), velocity_(Eigen::Vector3d::Zero()), 
      initialized_(false), velocity_filter_(filter_window) {}

Eigen::Vector3d VelocityEstimator::update(const Eigen::Vector3d& position) {
    if (!initialized_) {
        last_position_ = position;
        velocity_ = Eigen::Vector3d::Zero();
        initialized_ = true;
        return velocity_;
    }
    
    // Finite difference
    Eigen::Vector3d raw_velocity = (position - last_position_) / dt_;
    
    // Apply smoothing filter
    velocity_ = velocity_filter_.filter(raw_velocity);
    
    last_position_ = position;
    return velocity_;
}

void VelocityEstimator::reset() {
    initialized_ = false;
    velocity_ = Eigen::Vector3d::Zero();
    velocity_filter_.reset();
}

// OutlierFilter implementation
OutlierFilter::OutlierFilter(double max_jump, double max_vel)
    : max_jump_distance_(max_jump), max_velocity_(max_vel), initialized_(false) {}

bool OutlierFilter::is_valid(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity) {
    // Check velocity constraint
    if (velocity.norm() > max_velocity_) {
        return false;
    }
    
    // Check position jump
    if (initialized_) {
        double jump = (position - last_valid_position_).norm();
        if (jump > max_jump_distance_) {
            return false;
        }
    }
    
    return true;
}

void OutlierFilter::update_last_valid(const Eigen::Vector3d& position) {
    last_valid_position_ = position;
    initialized_ = true;
}

void OutlierFilter::reset() {
    initialized_ = false;
}

} // namespace optitrack_listener_cpp