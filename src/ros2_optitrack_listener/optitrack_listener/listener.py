# import rclpy
# from rclpy.node import Node
# from threading import Thread
# import optirx as rx
# import socket
# from geometry_msgs.msg import PoseStamped, Point, Quaternion
# import numpy as np

# class OptitrackListener(Node):
#     def __init__(self):
#         super().__init__('optitrack_listener')
        
#         # Network configuration
#         self.declare_parameter('local_interface', "172.31.1.145")
#         self.ipaddr = self.get_parameter('local_interface').get_parameter_value().string_value
        
#         # Frame configuration
#         self.declare_parameter('fixed_frame', 'world')
#         self.frame = self.get_parameter('fixed_frame').get_parameter_value().string_value
        
#         # Ball marker tracking parameters
#         self.declare_parameter('initial_ball_position', [0.0, 0.0, 0.0])
#         self.ball_marker_old = self.get_parameter('initial_ball_position').get_parameter_value().double_array_value
        
#         # EMA filtering parameters
#         self.declare_parameter('ball_ema_alpha', 0.5)
#         self.ema_alpha = self.get_parameter('ball_ema_alpha').get_parameter_value().double_value
#         self.declare_parameter('update_rate_hz', 20.0)
#         update_rate = self.get_parameter('update_rate_hz').get_parameter_value().double_value
#         self.dt = 1.0 / update_rate
        
#         # Initialize EMA state
#         self.ball_ema_position = np.array(self.ball_marker_old)
#         self.previous_ema = np.array(self.ball_marker_old)
#         self.ema_initialized = False
        
#         self.get_logger().info(f"EMA alpha: {self.ema_alpha}, Update rate: {update_rate} Hz")
        
#         # Publisher for ball marker
#         self.ball_marker_publisher = self.create_publisher(PoseStamped, "optitrack/ball_marker", 1)
#         self.ball_pose_stamped = PoseStamped()
        
#         # Start OptiTrack data thread
#         self.thread = Thread(target=self.get_optitrack_data, daemon=True)
#         self.thread.start()
    
#     def update_ema_position(self, new_position):
#         """Apply phase-compensated exponential moving average"""
#         new_pos_array = np.array(new_position)
        
#         if not self.ema_initialized:
#             self.ball_ema_position = new_pos_array.copy()
#             self.previous_ema = new_pos_array.copy()
#             self.ema_initialized = True
#             return self.ball_ema_position
        
#         # Standard EMA
#         ema_filtered = self.ema_alpha * new_pos_array + (1 - self.ema_alpha) * self.ball_ema_position
        
#         # Phase compensation
#         derivative = (ema_filtered - self.previous_ema) / self.dt
#         lag_time = (1 - self.ema_alpha) / self.ema_alpha * self.dt
#         compensated = ema_filtered + derivative * lag_time
        
#         # Update state
#         self.previous_ema = ema_filtered.copy()
#         self.ball_ema_position = ema_filtered.copy()
        
#         return compensated
    
#     def find_closest_marker(self, markers, reference):
#         """Find the marker closest to the reference position"""
#         if not markers:
#             return None
#         elif len(markers) == 1:
#             return markers[0]
        
#         # Use only x,z coordinates for horizontal plane matching
#         markers_xz = np.array(markers)[:, [0, 2]]
#         reference_xz = np.array(reference)[[0, 2]]
        
#         distances = np.linalg.norm(markers_xz - reference_xz, axis=1)
#         closest_index = np.argmin(distances)
        
#         return markers[closest_index]
    
#     def get_optitrack_data(self):
#         """Main data acquisition loop"""
#         version = (2, 7, 0, 0)
#         self.get_logger().info(f"Connecting to OptiTrack at {self.ipaddr}")
        
#         try:
#             optitrack_socket = rx.mkdatasock(ip_address=self.ipaddr)
#         except Exception as e:
#             self.get_logger().error(f"Failed to connect to OptiTrack: {e}")
#             return
        
#         first_packet = True
        
#         while rclpy.ok():
#             try:
#                 data = optitrack_socket.recv(rx.MAX_PACKETSIZE)
#                 packet = rx.unpack(data, version=version)
                
#                 if first_packet:
#                     self.get_logger().info(f"Connected! NatNet version: {version}")
#                     first_packet = False
                
#                 # Update version info if received
#                 if isinstance(packet, rx.SenderData):
#                     version = packet.natnet_version
#                     self.get_logger().info(f"NatNet version updated: {version}")
                
#                 # Process frame data
#                 if isinstance(packet, (rx.SenderData, rx.ModelDefs, rx.FrameOfData)):
#                     self.process_frame_data(packet)
                    
#             except socket.error as e:
#                 self.get_logger().warning(f"Socket error: {e}")
#             except Exception as e:
#                 self.get_logger().error(f"Unexpected error: {e}")
    
#     def process_frame_data(self, packet):
#         """Process incoming frame data and publish ball marker"""
#         # Update message header
#         self.ball_pose_stamped.header.stamp = self.get_clock().now().to_msg()
#         self.ball_pose_stamped.header.frame_id = self.frame
        
#         # Find closest marker to previous position
#         if hasattr(packet, 'other_markers'):
#             ball_marker = self.find_closest_marker(packet.other_markers, self.ball_marker_old)
            
#             if ball_marker is not None:
#                 # Apply EMA filtering
#                 ema_position = self.update_ema_position(ball_marker)
                
#                 # Update reference for next iteration (use raw measurement)
#                 self.ball_marker_old = ball_marker
                
#                 # Update pose message with filtered position
#                 self.ball_pose_stamped.pose.position = Point(
#                     x=float(ema_position[0]), 
#                     y=float(ema_position[1]), 
#                     z=float(ema_position[2])
#                 )
#                 self.ball_pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
#         # Publish the marker pose
#         self.ball_marker_publisher.publish(self.ball_pose_stamped)
#         # print(f"Published ball marker at {self.ball_pose_stamped.pose.position}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = OptitrackListener()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.node import Node
from threading import Thread
import optirx as rx
import socket
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion, Point
from tf2_ros import TransformBroadcaster
from tf2_geometry_msgs import do_transform_pose
import numpy as np
import re   

class OptitrackListener( Node ):
    def __init__(self):
        super().__init__('optitrack_listener')
        self.declare_parameter('local_interface',  "172.31.1.145")
        self.ipaddr = self.get_parameter('local_interface').get_parameter_value().string_value
        self.declare_parameter('fixed_frame', 'world')
        self.frame = self.get_parameter('fixed_frame').get_parameter_value().string_value
        self.declare_parameter('rigid_object_list', "")
        rigid_object_list = self.get_parameter('rigid_object_list').get_parameter_value().string_value
        split_string = rigid_object_list.split(', ')
        self.declare_parameter('publish_tf', False)
        self.publish_tf = self.get_parameter( 'publish_tf').get_parameter_value().bool_value
        names = rigid_object_list.split(',')
        self.get_logger().info("Trackable configuration {names}")

        self.id_trackable_dict = {}   # Dictionary: from object id to trackable name
        self.id_publisher_dict = {}   # Dictionary: from object id to publisher id 
        self.get_logger().info("Trackable configuration")
        self.trackable_publishers = []
        index = 0
        self.declare_parameter('initial_ball_position',  [0.0, 0.0, 0.0])
        self.ball_marker_old = self.get_parameter('initial_ball_position').get_parameter_value().double_array_value

        # EMA parameters
        self.declare_parameter('ball_ema_alpha', 1.0)
        self.ema_alpha = self.get_parameter('ball_ema_alpha').get_parameter_value().double_value
        
        # Initialize EMA position with the initial ball position
        self.ball_ema_position = np.array(self.ball_marker_old)
        self.ema_initialized = False
        
        self.get_logger().info(f"Ball EMA alpha value: {self.ema_alpha}")

        self.ball_marker_publisher = self.create_publisher(PoseStamped, "optitrack/ball_marker", 1)

        self.ball_pose_stamped = PoseStamped()

        for name in names:
            name = name.strip()
            self.declare_parameter(f'trackables.{name}.id', 0)
            self.declare_parameter(f'trackables.{name}.name', "trackable")
            id_param = self.get_parameter(f'trackables.{name}.id').get_parameter_value().integer_value
            name_param = self.get_parameter(f'trackables.{name}.name').get_parameter_value().string_value
            
            self.get_logger().info(f"Name: '{name}'")
            self.get_logger().info(f"id: {id_param}")
            self.get_logger().info(f"name: {name_param}")

            self.id_trackable_dict[id_param] = name_param
            self.id_publisher_dict[id_param] = index 

            self.trackable_publishers.append ( self.create_publisher(PoseStamped, "optitrack/" + name_param, 1))
            index = index+1

        self.tf_broadcaster = TransformBroadcaster(self)
        self.thread = Thread(target = self.get_optitrack_data, args = ())
        self.thread.start()

    def update_ema_position(self, new_position):
        """
        Update the exponential moving average of the ball position
        EMA formula: EMA_new = alpha * new_value + (1 - alpha) * EMA_old
        """
        new_pos_array = np.array(new_position)
        
        if not self.ema_initialized:
            # Initialize EMA with the first measurement
            self.ball_ema_position = new_pos_array.copy()
            self.ema_initialized = True
        else:
            # Update EMA
            self.ball_ema_position = self.ema_alpha * new_pos_array + (1 - self.ema_alpha) * self.ball_ema_position
        
        return self.ball_ema_position

    def pose_to_tf(self, pose_msg, parent_frame, target_frame):
        # Create a TransformStamped message
        transform_stamped = TransformStamped()
        
        # Set the header of the TF message
        transform_stamped.header.stamp = pose_msg.header.stamp
        transform_stamped.header.frame_id = parent_frame  
        transform_stamped.child_frame_id = target_frame  

        # Extract position and orientation from PoseStamped message
        position = pose_msg.pose.position
        orientation = pose_msg.pose.orientation
        
        # Set translation and rotation in the TF message
        transform_stamped.transform.translation.x = position.x
        transform_stamped.transform.translation.y = position.y
        transform_stamped.transform.translation.z = position.z
        transform_stamped.transform.rotation = orientation

        return transform_stamped  
    
    def find_closest_array_numpy(self, arrays, reference):
        
        if np.array(arrays).shape[0] == 0:
            return None
        elif np.array(arrays).shape[0] == 1:
            return arrays[0]
        else:
            arrays_np = np.array(arrays)[:, [0, 2]]
            reference_np = np.array(reference)[[0, 2]]  # Extract only first and third elements, because we are interested in x and z coordinates (optitrack frame horizontal plane)
            
            # Calculate all distances at once
            distances = np.linalg.norm(arrays_np - reference_np, axis=1)
            
            # Find index of minimum distance
            closest_index = np.argmin(distances)
            
            return arrays[closest_index]
    

    def get_optitrack_data(self):

        
        self.version = (2, 7, 0, 0)  # the latest SDK version
        self.get_logger().warn(f"address: {self.ipaddr}")

        self.optitrack = rx.mkdatasock(ip_address=self.ipaddr)#(ip_address=get_ip_address(iface))
              
        ps = PoseStamped()

        first = True
        # self.get_logger().info(f"here1111")
        # print("here3333")
        while rclpy.ok():
            # print("here4444")
            try:
                data = self.optitrack.recv(rx.MAX_PACKETSIZE)
            except socket.error:
                self.get_logger().info(f"Failed to receive packet from optitrack")

            packet = rx.unpack(data, version=self.version)
            # print("packet", packet)
            if first == True:
                self.get_logger().info(f"NatNet version received {self.version}")
                first = False
            
            if type(packet) is rx.SenderData:
                self.version = packet.natnet_version
                self.get_logger().info(f"NatNet version received {self.version}")

            if type(packet) in [rx.SenderData, rx.ModelDefs, rx.FrameOfData]:
                #Fill message header
                self.ball_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                self.ball_pose_stamped.header.frame_id = self.frame
                # print(self.ball_pose_stamped.pose.position)
                
                #Find the ball marker (closest to the old position, initialized with a known configuration)
                ball_marker = self.find_closest_array_numpy(packet.other_markers, self.ball_marker_old)
                # print(f"Ball marker found: {ball_marker}")
                
                if ball_marker is not None: #Position is updated only if at least a marker is found, otherwise the old position is kept
                    # print(f"Ball marker found: {ball_marker}")
                    # Update EMA position with the new measurement
                    ema_position = self.update_ema_position(ball_marker)
                    
                    # Update the old position for closest marker search (use raw measurement)
                    self.ball_marker_old = ball_marker
                    
                    # Fill the message with EMA-smoothed position
                    # self.ball_pose_stamped.pose.position = Point(x=ball_marker[0], y=ball_marker[1], z=ball_marker[2])
                    self.ball_pose_stamped.pose.position = Point(x=ema_position[0], y=ema_position[1], z=ema_position[2])
                    self.ball_pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                
                # Publish the ball marker (with EMA-smoothed position)
                self.ball_marker_publisher.publish(self.ball_pose_stamped)
                    
def main(args=None):
    rclpy.init(args=args)
    node = OptitrackListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()