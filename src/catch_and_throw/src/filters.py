class RealTimeSavitzkyGolay:
    """Real-time Savitzky-Golay filter implementation for smoothing 3D position data"""
    
    def __init__(self, window_length=7, polyorder=2, deriv=0, delta=1.0, initial_values=None):
        """
        Initialize real-time Savitzky-Golay filter
        
        Args:
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial fit
            deriv: Derivative order (0 for position, 1 for velocity, etc.)
            delta: Sample spacing (for derivative calculation)
            initial_values: Initial buffer values (optional)
        """
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
            print(f"Warning: window_length adjusted to odd value: {window_length}")
        
        # Ensure polyorder is valid
        if polyorder >= window_length:
            polyorder = window_length - 1
            print(f"Warning: polyorder too large, reduced to {polyorder}")
        
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        
        # Compute Savitzky-Golay coefficients
        # For real-time filtering, we need coefficients for the last point in the window
        self.coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta, pos=window_length-1)
        
        # Initialize circular buffer
        if initial_values is not None:
            if isinstance(initial_values, np.ndarray):
                if initial_values.ndim == 1:
                    # 1D data
                    if len(initial_values) >= window_length:
                        self.buffer = initial_values[-window_length:].copy()
                    else:
                        # Pad with first value
                        padding = np.full(window_length - len(initial_values), initial_values[0])
                        self.buffer = np.concatenate([padding, initial_values])
                else:
                    # Multi-dimensional data
                    if initial_values.shape[0] >= window_length:
                        self.buffer = initial_values[-window_length:].copy()
                    else:
                        # Pad with first value
                        padding = np.repeat(initial_values[0:1], window_length - initial_values.shape[0], axis=0)
                        self.buffer = np.concatenate([padding, initial_values], axis=0)
            else:
                self.buffer = np.zeros((window_length, 3))  # Default for 3D position
                self.buffer_initialized = False
        else:
            self.buffer = np.zeros((window_length, 3))  # Default for 3D position
            self.buffer_initialized = False
        
        self.current_idx = 0
        self.buffer_initialized = initial_values is not None
    
    def __call__(self, x):
        """
        Apply Savitzky-Golay filter to new data point
        
        Args:
            x: New data point (can be scalar or array)
            
        Returns:
            Filtered value
        """
        # Convert to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        # Initialize buffer shape on first real data
        if not self.buffer_initialized:
            if x.ndim == 0:
                self.buffer = np.zeros(self.window_length)
            else:
                self.buffer = np.zeros((self.window_length, *x.shape))
            self.buffer_initialized = True
        
        # Store new value in circular buffer
        self.buffer[self.current_idx] = x
        self.current_idx = (self.current_idx + 1) % self.window_length
        
        # Reorder buffer to have oldest to newest
        if self.current_idx == 0:
            ordered_buffer = self.buffer
        else:
            ordered_buffer = np.concatenate([
                self.buffer[self.current_idx:],
                self.buffer[:self.current_idx]
            ])
        
        # Apply Savitzky-Golay coefficients
        if ordered_buffer.ndim == 1:
            # 1D data
            filtered_value = np.dot(ordered_buffer, self.coeffs)
        else:
            # Multi-dimensional data - apply filter to each dimension
            filtered_value = np.zeros_like(x)
            for i in range(ordered_buffer.shape[1]):
                filtered_value[i] = np.dot(ordered_buffer[:, i], self.coeffs)
        
        return filtered_value
    
    def reset(self):
        """Reset the filter buffer"""
        self.buffer[:] = 0
        self.current_idx = 0
        self.buffer_initialized = False


class VelocityEstimator:
    """Estimate velocity using finite differences with smoothing"""
    
    def __init__(self, dt=0.005, alpha=0.9):
        """
        Initialize velocity estimator
        
        Args:
            dt: Time step
            alpha: Smoothing factor (0-1, higher = more smoothing)
        """
        self.dt = dt
        self.alpha = alpha
        self.last_position = None
        self.velocity = np.zeros(3)
        self.initialized = False
    
    def update(self, position):
        """Update velocity estimate"""
        if not self.initialized:
            self.last_position = position.copy()
            self.velocity = np.zeros(3)
            self.initialized = True
            return self.velocity
        
        # Calculate instantaneous velocity
        instant_velocity = (position - self.last_position) / self.dt
        
        # Apply exponential smoothing
        self.velocity = self.alpha * self.velocity + (1 - self.alpha) * instant_velocity
        
        # Update last position
        self.last_position = position.copy()
        
        return self.velocity.copy()
    
    def reset(self):
        """Reset the velocity estimator"""
        self.last_position = None
        self.velocity = np.zeros(3)
        self.initialized = False

