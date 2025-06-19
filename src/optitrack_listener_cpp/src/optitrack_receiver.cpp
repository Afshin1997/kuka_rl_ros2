#include "optitrack_listener_cpp/optitrack_receiver.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

namespace optitrack_listener_cpp {

template<typename T>
T OptiTrackReceiver::read_value(const char*& ptr) {
    T value = *reinterpret_cast<const T*>(ptr);
    ptr += sizeof(T);
    return value;
}

std::string OptiTrackReceiver::read_string(const char*& ptr) {
    std::string str(ptr);
    ptr += str.length() + 1;
    return str;
}

OptiTrackReceiver::OptiTrackReceiver(const std::string& multicast_addr, int port)
    : socket_fd_(-1), connected_(false), multicast_address_(multicast_addr), 
      port_(port), nat_net_major_(2), nat_net_minor_(7) {}

OptiTrackReceiver::~OptiTrackReceiver() {
    disconnect();
}

bool OptiTrackReceiver::connect() {
    if (connected_) {
        return true;
    }
    
    // Create UDP socket
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        return false;
    }
    
    // Allow multiple sockets to use the same PORT number
    int reuse = 1;
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    // Set up destination address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    addr.sin_addr.s_addr = INADDR_ANY;
    
    // Bind to receive address
    if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    // Join multicast group
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(multicast_address_.c_str());
    mreq.imr_interface.s_addr = INADDR_ANY;
    
    if (setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    // Set receive timeout
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 1000; // 1ms timeout for responsiveness
    setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    
    connected_ = true;
    return true;
}

void OptiTrackReceiver::disconnect() {
    if (connected_ && socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
        connected_ = false;
    }
}

bool OptiTrackReceiver::receive_packet(OptiTrackPacket& packet) {
    if (!connected_) {
        return false;
    }
    
    const size_t MAX_PACKET_SIZE = 65507;
    char buffer[MAX_PACKET_SIZE];
    
    ssize_t bytes_received = recv(socket_fd_, buffer, MAX_PACKET_SIZE, 0);
    
    if (bytes_received <= 4) {
        return false;
    }
    
    const char* ptr = buffer;
    
    // Read message ID and packet size
    uint16_t message_id = read_value<uint16_t>(ptr);
    uint16_t packet_size = read_value<uint16_t>(ptr);
    
    if (packet_size != bytes_received - 4) {
        return false;
    }
    
    // Parse based on message type
    switch (message_id) {
        case OptiTrackPacket::SENDER_DATA:
            packet.type = OptiTrackPacket::SENDER_DATA;
            return parse_sender_data(ptr, packet_size, packet);
            
        case OptiTrackPacket::FRAME_DATA:
            packet.type = OptiTrackPacket::FRAME_DATA;
            return parse_frame_data(ptr, packet_size, packet);
            
        default:
            return false;
    }
}

bool OptiTrackReceiver::parse_sender_data(const char* data, size_t size, OptiTrackPacket& packet) {
    const char* ptr = data;
    const char* end = data + size;
    
    if (ptr + 256 > end) return false;
    
    // Skip application name
    ptr += 256;
    
    // Skip version info
    if (ptr + 4 * sizeof(uint8_t) > end) return false;
    ptr += 4;
    
    // Read NatNet version
    if (ptr + 4 * sizeof(uint8_t) > end) return false;
    nat_net_major_ = read_value<uint8_t>(ptr);
    nat_net_minor_ = read_value<uint8_t>(ptr);
    ptr += 2; // Skip build numbers
    
    return true;
}

bool OptiTrackReceiver::parse_frame_data(const char* data, size_t size, OptiTrackPacket& packet) {
    const char* ptr = data;
    const char* end = data + size;
    
    try {
        // Frame number
        if (ptr + sizeof(uint32_t) > end) return false;
        packet.frame_number = read_value<uint32_t>(ptr);
        
        // Skip marker sets
        if (ptr + sizeof(uint32_t) > end) return false;
        uint32_t num_marker_sets = read_value<uint32_t>(ptr);
        
        for (uint32_t i = 0; i < num_marker_sets; i++) {
            if (ptr >= end) return false;
            read_string(ptr); // Skip name
            
            if (ptr + sizeof(uint32_t) > end) return false;
            uint32_t num_markers = read_value<uint32_t>(ptr);
            
            // Skip marker positions
            size_t marker_data_size = num_markers * 3 * sizeof(float);
            if (ptr + marker_data_size > end) return false;
            ptr += marker_data_size;
        }
        
        // Skip unlabeled markers
        if (ptr + sizeof(uint32_t) > end) return false;
        uint32_t num_unlabeled = read_value<uint32_t>(ptr);
        size_t unlabeled_size = num_unlabeled * 3 * sizeof(float);
        if (ptr + unlabeled_size > end) return false;
        ptr += unlabeled_size;
        
        // Parse rigid bodies
        if (ptr + sizeof(uint32_t) > end) return false;
        uint32_t num_rigid_bodies = read_value<uint32_t>(ptr);
        
        packet.rigid_bodies.clear();
        packet.rigid_bodies.reserve(num_rigid_bodies);
        
        for (uint32_t i = 0; i < num_rigid_bodies; i++) {
            RigidBodyData rb;
            
            // ID
            if (ptr + sizeof(uint32_t) > end) return false;
            rb.id = read_value<uint32_t>(ptr);
            
            // Position
            if (ptr + 3 * sizeof(float) > end) return false;
            float x = read_value<float>(ptr);
            float y = read_value<float>(ptr);
            float z = read_value<float>(ptr);
            rb.position = {x, y, z};
            
            // Orientation (quaternion)
            if (ptr + 4 * sizeof(float) > end) return false;
            float qx = read_value<float>(ptr);
            float qy = read_value<float>(ptr);
            float qz = read_value<float>(ptr);
            float qw = read_value<float>(ptr);
            rb.orientation = {qx, qy, qz, qw};
            
            // Skip associated marker data
            if (ptr + sizeof(uint32_t) > end) return false;
            uint32_t num_rb_markers = read_value<uint32_t>(ptr);
            
            // Skip marker IDs
            size_t marker_id_size = num_rb_markers * sizeof(uint32_t);
            if (ptr + marker_id_size > end) return false;
            ptr += marker_id_size;
            
            // Skip marker sizes
            size_t marker_size_size = num_rb_markers * sizeof(float);
            if (ptr + marker_size_size > end) return false;
            ptr += marker_size_size;
            
            // Mean marker error (tracking valid indicator)
            if (ptr + sizeof(float) > end) return false;
            rb.tracking_valid = read_value<float>(ptr);
            
            // NatNet 2.6+ has tracking flags
            if (nat_net_major_ >= 2 && nat_net_minor_ >= 6) {
                if (ptr + sizeof(uint16_t) > end) return false;
                uint16_t tracking_flags = read_value<uint16_t>(ptr);
                // Bit 0 indicates tracking status
                rb.tracking_valid = (tracking_flags & 0x01) ? 1.0f : 0.0f;
            }
            
            packet.rigid_bodies.push_back(rb);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

} // namespace optitrack_listener_cpp