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
import csv

class OptitrackListener( Node ):
    def __init__(self):
        super().__init__('optitrack_listener')
        self.declare_parameter('local_interface',  "192.168.1.100")
        self.ipaddr = self.get_parameter('local_interface').get_parameter_value().string_value
        self.declare_parameter('fixed_frame', 'world')
        self.frame = self.get_parameter('fixed_frame').get_parameter_value().string_value
        self.declare_parameter('rigid_object_list', "")
        rigid_object_list = self.get_parameter('rigid_object_list').get_parameter_value().string_value
        split_string = rigid_object_list.split(', ')
        self.declare_parameter('publish_tf', False)

        ###
        self.declare_parameter('csv_file_path', 'optitrack_data.csv')
        self.csv_file_path = self.get_parameter('csv_file_path').value
        ###

        self.publish_tf = self.get_parameter( 'publish_tf').get_parameter_value().bool_value
        names = rigid_object_list.split(',')

        ### Initialize CSV writing components
        self.csv_file = None
        self.csv_writer = None
        self.csv_initialized = False
        ###

        self.get_logger().info("Trackable configuration {names}")

        self.id_trackable_dict = {}   # Dictionary: from object id to trackable name
        self.id_publisher_dict = {}   # Dictionary: from object id to publisher id 
        self.get_logger().info("Trackable configuration")
        self.trackable_publishers = []
        index = 0

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

    def initialize_csv(self):
        try:
            self.csv_file = open(self.csv_file_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'body_id', 'x', 'y', 'z'])
            self.csv_initialized = True
            self.get_logger().info(f"CSV logging initialized at {self.csv_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize CSV: {str(e)}")

    def get_optitrack_data(self):
        
        self.version = (2, 7, 0, 0)  # the latest SDK version
        self.get_logger().warn(f"address: {self.ipaddr}")

        self.optitrack = rx.mkdatasock(ip_address=self.ipaddr)#(ip_address=get_ip_address(iface))
        
        # Initialize CSV
        self.initialize_csv()

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
                for i, rigid_body in enumerate(packet.rigid_bodies):
                    body_id = rigid_body.id
                    pos_opt = np.array(rigid_body.position)
                    rot_opt = np.array(rigid_body.orientation)
                    
                    ps.header.stamp = self.get_clock().now().to_msg()  
                    ps.header.frame_id = self.frame
                    ps.pose.position = Point(x=pos_opt[0], y=pos_opt[1], z=pos_opt[2])  
                    ps.pose.orientation = Quaternion(x=rot_opt[0], y=rot_opt[1], z=rot_opt[2], w=rot_opt[3])  


                    if self.csv_initialized:
                        try:
                            timestamp = ps.header.stamp.sec + ps.header.stamp.nanosec / 1e9
                            if pos_opt[2] <= 1.0 and pos_opt[2] >= -3.5 and pos_opt[1] >= 0.5:
                                self.csv_writer.writerow([
                                    timestamp,
                                    body_id,
                                    pos_opt[0],
                                    pos_opt[1],
                                    pos_opt[2]
                                ])
                                self.csv_file.flush()
                        except Exception as e:
                            self.get_logger().error(f"CSV write error: {str(e)}")

                        finally:
                            if self.csv_file is not None:
                                self.csv_file.close()
                                self.get_logger().info("CSV file closed")


                    if body_id in self.id_publisher_dict:
                        self.trackable_publishers[ self.id_publisher_dict[body_id] ].publish( ps )
                        if( self.publish_tf):
                            transform_stamped = self.pose_to_tf(ps, self.frame, self.id_trackable_dict[body_id] )                    
                            self.tf_broadcaster.sendTransform(transform_stamped)
                    else: 
                        self.get_logger().warn(f"Id: {body_id} not present in the configuration file")
        
                    

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
