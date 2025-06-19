#ifndef OPTITRACK_LISTENER_CPP_OPTITRACK_RECEIVER_HPP
#define OPTITRACK_LISTENER_CPP_OPTITRACK_RECEIVER_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <array>

namespace optitrack_listener_cpp {

// Structure to hold rigid body data
struct RigidBodyData {
    int id;
    std::array<double, 3> position;      // x, y, z
    std::array<double, 4> orientation;   // x, y, z, w (quaternion)
    float tracking_valid;
};

// OptiTrack packet structure
struct OptiTrackPacket {
    enum Type { 
        SENDER_DATA = 5,
        FRAME_DATA = 7 
    };
    
    Type type;
    uint32_t frame_number;
    std::vector<RigidBodyData> rigid_bodies;
};

// UDP receiver for OptiTrack NatNet protocol
class OptiTrackReceiver {
private:
    int socket_fd_;
    bool connected_;
    std::string multicast_address_;
    int port_;
    
    // NatNet version
    uint8_t nat_net_major_;
    uint8_t nat_net_minor_;
    
    // Helper functions for parsing
    template<typename T>
    T read_value(const char*& ptr);
    
    std::string read_string(const char*& ptr);
    
    bool parse_sender_data(const char* data, size_t size, OptiTrackPacket& packet);
    bool parse_frame_data(const char* data, size_t size, OptiTrackPacket& packet);
    
public:
    OptiTrackReceiver(const std::string& multicast_addr = "239.255.42.99", 
                     int port = 1511);
    ~OptiTrackReceiver();
    
    bool connect();
    void disconnect();
    bool is_connected() const { return connected_; }
    
    bool receive_packet(OptiTrackPacket& packet);
    
    void set_version(uint8_t major, uint8_t minor) {
        nat_net_major_ = major;
        nat_net_minor_ = minor;
    }
};

} // namespace optitrack_listener_cpp

#endif // OPTITRACK_LISTENER_CPP_OPTITRACK_RECEIVER_HPP