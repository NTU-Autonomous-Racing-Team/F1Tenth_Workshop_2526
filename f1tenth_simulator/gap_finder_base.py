#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray

class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')
 
        self.declare_parameter('lookahead_distance', 5.0) 
        self.declare_parameter('robot_width', 0.2032)
        self.declare_parameter('obstacle_bubble_radius', 0.15) 
        self.declare_parameter('disparity_threshold', 0.5) 
        self.declare_parameter('max_speed', 5.0) #10.0 
        self.declare_parameter('min_speed', 1.0)
        self.declare_parameter('max_steering', 0.34) 
        self.declare_parameter('disparity_bubble_radius', 0.15)
        self.declare_parameter('consecutive_valid_gap', 5)
        self.declare_parameter('steering_gain', 0.4) 
        self.declare_parameter('speed_gain', 1.0) 
        self.declare_parameter('field_of_vision', np.pi/2)
        self.declare_parameter('lidarscan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
            
        self.lookahead_distance     = self.get_parameter('lookahead_distance').value
        self.robot_width            = self.get_parameter('robot_width').value
        self.obstacle_bubble_radius = self.get_parameter('obstacle_bubble_radius').value
        self.disparity_threshold    = self.get_parameter('disparity_threshold').value
        self.max_speed              = self.get_parameter('max_speed').value
        self.min_speed              = self.get_parameter('min_speed').value
        self.max_steering           = self.get_parameter('max_steering').value
        self.disparity_bubble_radius= self.get_parameter('disparity_bubble_radius').value
        self.consecutive_valid_gap  = self.get_parameter('consecutive_valid_gap').value
        self.steering_gain          = self.get_parameter('steering_gain').value
        self.speed_gain             = self.get_parameter('speed_gain').value
        self.field_of_vision        = self.get_parameter('field_of_vision').value
        self.lidar_scan_topic       = self.get_parameter('lidarscan_topic').value
        self.drive_topic            = self.get_parameter('drive_topic').value
            
        self.subscriber = self.create_subscription(LaserScan, 
                                                   self.lidar_scan_topic,
                                                   self.lidar_callback,
                                                   10)
        
        self.publisher = self.create_publisher(AckermannDriveStamped,
                                                self.drive_topic,
                                                10)

        self.bubble_viz_publisher = self.create_publisher(MarkerArray, "/safety_bubble", 10)
        self.scan_viz_publisher = self.create_publisher(MarkerArray, "/scan_msg", 10)
        self.gap_viz_publisher = self.create_publisher(Marker, "/goal_point", 10)
        self.dispa_viz_publisher = self.create_publisher(MarkerArray, "/disparity_points", 10)

    def preprocess_lidar(self,data):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # convert the FOV into indices
        ranges = np.array(data.ranges)
        fov_indices = int(self.field_of_vision / data.angle_increment)
        center_index = int(abs(data.angle_min) / data.angle_increment)  # // will return float, e.g. 540.0 if input is float value
        fov_start_i = center_index - fov_indices
        fov_end_i = center_index + fov_indices

        # handle edge cases
        fov_start_i = max(0, fov_start_i)
        fov_end_i = min(len(ranges)-1, fov_end_i)

        limited_ranges = np.copy(ranges)  # Alt: Truncate the ranges array
        limited_ranges[:fov_start_i] = 0
        limited_ranges[fov_end_i+1:] = 0

        # find nearest obstacle distance & index
        nearest_obstacle_distance = np.min(limited_ranges[limited_ranges > 0])
        nearest_obstacle_distance_indices = np.where(limited_ranges == nearest_obstacle_distance)

        # find the angle and indices span of the obstacle bubble 
        obs_bubble_half_angle = np.arctan(self.obstacle_bubble_radius / nearest_obstacle_distance)
        obs_bubble_index_extension = int(np.ceil(obs_bubble_half_angle / data.angle_increment))

        obs_bubble_start_i = np.maximum(0, nearest_obstacle_distance_indices[0] - obs_bubble_index_extension)
        obs_bubble_end_i = np.minimum(len(limited_ranges)-1, nearest_obstacle_distance_indices[0] + obs_bubble_index_extension)
        valid_limited_ranges = np.copy(limited_ranges)

        # draw safety bubble
        for start, end in zip(obs_bubble_start_i, obs_bubble_end_i):
            valid_limited_ranges[start:end] = 0

        proc_ranges = np.copy(valid_limited_ranges)

        # find bubble_coord
        nearest_obstacle_angle = data.angle_min + nearest_obstacle_distance_indices[0] * data.angle_increment
        obs_x = nearest_obstacle_distance * np.cos(nearest_obstacle_angle)
        obs_y = nearest_obstacle_distance * np.sin(nearest_obstacle_angle)

        # Turns [[x1, x2], [y1, y2]] into [[x1, y1], [x2, y2]]
        obs_bubble_coord = np.array([obs_x,obs_y]).T

        return proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i

    def find_max_gap(self, data, proc_ranges):
        """ Return the start index & end index of the max gap in proc_ranges
        """
        # find the furthest point in the processed ranges array
        max_dist = np.max(proc_ranges)
        max_dist_indices = np.where(proc_ranges == max_dist)[0]

        if len(max_dist_indices) > 1:
            max_dist_index = max_dist_indices[len(max_dist_indices)//2]
        else:
            max_dist_index = max_dist_indices[0]
        
        # find goal coord
        goal_distance = proc_ranges[max_dist_index]
        goal_angle = data.angle_min + max_dist_index * data.angle_increment
        goal_x = goal_distance * np.cos(goal_angle)
        goal_y = goal_distance * np.sin(goal_angle)
        goal_coord = np.array([goal_x, goal_y])
    
        return max_dist_index, goal_coord

    def disparity_extender(self, data):
        proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i = self.preprocess_lidar(data)

        # find the difference between elements n+1 and n
        range_diffs = np.abs(np.diff(proc_ranges))
        disparity_indices = np.where(range_diffs > self.disparity_threshold)[0]

        if len(disparity_indices) == 0:
            proc_ranges = np.where(proc_ranges > self.lookahead_distance, self.lookahead_distance, proc_ranges)
            return proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i, []
        
        # '>0.01': avoid mistreat the index next to the 'zero' we created ourselves as a disparity
        valid_indices_mask = (proc_ranges[disparity_indices] > 0.01) & (proc_ranges[disparity_indices + 1] > 0.01)
        # e.g., VALID INDICES MASK: [False  True  True  True  True  True  True  True  True  True  True  True]
        disparity_indices = disparity_indices[valid_indices_mask] 

        if len(disparity_indices) == 0:
            proc_ranges = np.where(proc_ranges > self.lookahead_distance, self.lookahead_distance, proc_ranges)
            return proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i, []

        
        # find disparity coord
        disparity_angles = data.angle_min + disparity_indices * data.angle_increment
        disparity_x = proc_ranges[disparity_indices] * np.cos(disparity_angles) 
        disparity_y = proc_ranges[disparity_indices] * np.sin(disparity_angles) 
        disparity_coord = np.column_stack((disparity_x, disparity_y))
        
        # draw disparity bubble
        # Ensure the bubble drawn is always based on the closer distance
        p1 = proc_ranges[disparity_indices]
        p2 = proc_ranges[disparity_indices + 1] # confirm not out of bounds as index from |135deg| to |90deg| has alr been set to 0
        dist_for_calculation = np.minimum(p1, p2)

        dispa_bubble_half_angle = np.arctan(self.disparity_bubble_radius / dist_for_calculation)
        dispa_bubble_index_extension = (dispa_bubble_half_angle / data.angle_increment).astype(int)
        dispa_bubble_start_i = np.maximum(0, disparity_indices - dispa_bubble_index_extension)
        dispa_bubble_end_i = np.minimum(len(proc_ranges)-1,disparity_indices + dispa_bubble_index_extension)

        for start, end in zip(dispa_bubble_start_i, dispa_bubble_end_i):
            proc_ranges[start:end] = 0.0

        # cap vehicle vision
        proc_ranges = np.where(proc_ranges > self.lookahead_distance, self.lookahead_distance, proc_ranges)

        return proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i, disparity_coord

    def get_ranges_coord(self, data, proc_ranges, fov_start_i ,fov_end_i):
        indices = np.arange(len(proc_ranges[fov_start_i:fov_end_i+1]))
        angles = -np.pi/2 + (indices * data.angle_increment)

        x_coords = proc_ranges[fov_start_i:fov_end_i+1] * np.cos(angles)
        y_coords = proc_ranges[fov_start_i:fov_end_i+1] * np.sin(angles)

        ranges_coord = np.column_stack((x_coords, y_coords))
        # both .T and .column_stack work        
        return ranges_coord

    
    def visualisation_marker(self, bubble_coord, goal_coord, scan_msg_coord, dispa_coord):
        bubble_array_viz_msg = MarkerArray()
        for i, coord in enumerate(bubble_coord):
            self.bubble_viz_msg = Marker()
            self.bubble_viz_msg.header.frame_id = "ego_racecar/base_link"
            self.bubble_viz_msg.color.a = 1.0
            self.bubble_viz_msg.color.r = 1.0
            self.bubble_viz_msg.scale.x = self.obstacle_bubble_radius
            self.bubble_viz_msg.scale.y = self.obstacle_bubble_radius
            self.bubble_viz_msg.scale.z = self.obstacle_bubble_radius
            self.bubble_viz_msg.type = Marker.SPHERE
            self.bubble_viz_msg.action = Marker.ADD
            self.bubble_viz_msg.id = i
            self.bubble_viz_msg.pose.position.x = coord[0]
            self.bubble_viz_msg.pose.position.y = coord[1]
            bubble_array_viz_msg.markers.append(self.bubble_viz_msg)

        scan_array_viz_msg = MarkerArray()
        for i, coord in enumerate(scan_msg_coord):
            self.scan_viz_msg = Marker()
            self.scan_viz_msg.header.frame_id = "ego_racecar/base_link"
            self.scan_viz_msg.color.b = 0.95
            self.scan_viz_msg.color.a = 0.5
            self.scan_viz_msg.scale.x = 0.2
            self.scan_viz_msg.scale.y = 0.2
            self.scan_viz_msg.scale.z = 0.2
            self.scan_viz_msg.type = Marker.CYLINDER
            self.scan_viz_msg.action = Marker.ADD
            self.scan_viz_msg.id = i
            self.scan_viz_msg.pose.position.x = coord[0]
            self.scan_viz_msg.pose.position.y = coord[1]
            scan_array_viz_msg.markers.append(self.scan_viz_msg)

        dispa_array_viz_msg = MarkerArray()
        for i, coord in enumerate(dispa_coord):
            self.dispa_viz_msg = Marker()
            self.dispa_viz_msg.header.frame_id = "ego_racecar/base_link"
            self.dispa_viz_msg.color.r = 0.95
            self.dispa_viz_msg.color.g = 0.7
            self.dispa_viz_msg.color.b = 0.83
            self.dispa_viz_msg.color.a = 0.8
            self.dispa_viz_msg.scale.x = self.disparity_bubble_radius
            self.dispa_viz_msg.scale.y = self.disparity_bubble_radius
            self.dispa_viz_msg.scale.z = self.disparity_bubble_radius
            self.dispa_viz_msg.type = Marker.SPHERE
            self.dispa_viz_msg.action = Marker.ADD
            self.dispa_viz_msg.id = i
            self.dispa_viz_msg.pose.position.x = coord[0]
            self.dispa_viz_msg.pose.position.y = coord[1]
            dispa_array_viz_msg.markers.append(self.dispa_viz_msg)

        self.goal_viz_msg = Marker()
        self.goal_viz_msg.header.frame_id = "ego_racecar/base_link"
        self.goal_viz_msg.color.r = 0.3
        self.goal_viz_msg.color.g = 0.7
        self.goal_viz_msg.color.b = 0.0
        self.goal_viz_msg.color.a = 0.8
        self.goal_viz_msg.scale.x = 0.3
        self.goal_viz_msg.scale.y = 0.3
        self.goal_viz_msg.scale.z = 0.3
        self.goal_viz_msg.type = Marker.CYLINDER
        self.goal_viz_msg.action = Marker.ADD
        self.goal_viz_msg.pose.position.x = float(goal_coord[0])
        self.goal_viz_msg.pose.position.y = float(goal_coord[1])
        
        self.bubble_viz_publisher.publish(bubble_array_viz_msg)
        self.gap_viz_publisher.publish(self.goal_viz_msg)
        self.scan_viz_publisher.publish(scan_array_viz_msg)
        self.dispa_viz_publisher.publish(dispa_array_viz_msg)
 
    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i, dispa_coord = self.disparity_extender(data)
        max_dist_index, goal_coord = self.find_max_gap(data, proc_ranges)
        ranges_coord = self.get_ranges_coord(data, proc_ranges, fov_start_i, fov_end_i)
        
        msg = AckermannDriveStamped()
        steering_angle = 0.0
        if max_dist_index is not None:
            steering_angle = data.angle_min + max_dist_index * data.angle_increment
            steering_angle = np.clip(steering_angle * self.steering_gain, -self.max_steering, self.max_steering)
            msg.drive.steering_angle = steering_angle
        
        raw_speed = self.max_speed - abs(steering_angle) * self.speed_gain
        msg.drive.speed = max(self.min_speed, raw_speed)

        self.visualisation_marker(obs_bubble_coord, goal_coord, ranges_coord, dispa_coord)

        self.get_logger().info(f'steering_angle: {msg.drive.steering_angle}, speed: {msg.drive.speed}') 

        self.publisher.publish(msg)
        print("Published Drive Message")

def main(args=None):
    rclpy.init(args=args)
    reactive_node = GapFollower()
    print("I am blind, I can't find the gap :(")
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
