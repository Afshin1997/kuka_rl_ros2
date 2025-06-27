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
        self.declare_parameter('local_interface',  "192.168.1.4")
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

        self.ball_marker_publisher = self.create_publisher(PoseStamped, "optitrack/ball_marker", 1)

        self.ball_pose_stamped = PoseStamped()


        # self.declare_parameter('frame_transformation',  [0.0, 0.0, 0.0])
        # self.frame_transformation = self.get_parameter('frame_transformation').get_parameter_value().double_array_value

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
        while rclpy.ok():
            try:
                data = self.optitrack.recv(rx.MAX_PACKETSIZE)
            except socket.error:
                self.get_logger().info(f"Failed to receive packet from optitrack")

            packet = rx.unpack(data, version=self.version)
            if first == True:
                self.get_logger().info(f"NatNet version received {self.version}")
                first = False
            
            if type(packet) is rx.SenderData:
                self.version = packet.natnet_version
                self.get_logger().info(f"NatNet version received {self.version}")

            if type(packet) in [rx.SenderData, rx.ModelDefs, rx.FrameOfData]:
                # for i, rigid_body in enumerate(packet.rigid_bodies):
                #     body_id = rigid_body.id
                #     pos_opt = np.array(rigid_body.position)
                #     rot_opt = np.array(rigid_body.orientation)
                    
                #     ps.header.stamp = self.get_clock().now().to_msg()  
                #     ps.header.frame_id = self.frame
                #     ps.pose.position = Point(x=pos_opt[0], y=pos_opt[1], z=pos_opt[2])  
                #     ps.pose.orientation = Quaternion(x=rot_opt[0], y=rot_opt[1], z=rot_opt[2], w=rot_opt[3])  


                #     if body_id in self.id_publisher_dict:
                #         self.trackable_publishers[ self.id_publisher_dict[body_id] ].publish( ps )
                #         if( self.publish_tf):
                #             transform_stamped = self.pose_to_tf(ps, self.frame, self.id_trackable_dict[body_id] )                    
                #             self.tf_broadcaster.sendTransform(transform_stamped)
                #     else: 
                #         self.get_logger().warn(f"Id: {body_id} not present in the configuration file")

                #Fill message header
                self.ball_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                self.ball_pose_stamped.header.frame_id = self.frame
                
                #Find the ball marker (closest to the old position, initialized with a known configuration)
                ball_marker=self.find_closest_array_numpy(packet.other_markers, self.ball_marker_old)
                
                if ball_marker is not None: #Position is updated only if at least a marker is found, otherwise the old position is kept
                    #update the old position
                    self.ball_marker_old = ball_marker
                    #Fill the message (just position, orientation is not used)
                    self.ball_pose_stamped.pose.position = Point(x=ball_marker[0], y=ball_marker[1], z=ball_marker[2])
                    self.ball_pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                
                #Publish the ball marker
                self.ball_marker_publisher.publish(self.ball_pose_stamped)
                    
                #Add here tf if needed ...

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
