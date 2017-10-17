#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.initialized = False
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.lights_closest_wp = []
        self.stop_lines = []
        self.stop_lines_closest_wp = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic 
        light in 3D map space and helps you acquire an accurate ground truth 
        data source for the traffic light classifier by sending the current 
        color state of all traffic lights in the simulator. When testing on the 
        vehicle, the color state will not be available. You'll need to rely on 
        the position of the light and the camera image to predict it.
        '''
        self.sub3 = rospy.Subscriber('/vehicle/traffic_lights', 
            TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', 
            Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.initialized = True
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.sub2.unregister()

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if (self.waypoints is not None) and (len(self.stop_lines) == 0):
            stops = self.config['stop_line_positions']

            for light in self.lights:
                light_pose = light.pose.pose
                self.lights_closest_wp.append(self.get_closest_waypoint(light_pose))

            for stop in stops:
                stop_line_pose = Pose()
                stop_line_pose.position = Point()
                stop_line_pose.position.x = stop[0]
                stop_line_pose.position.y = stop[1]
                self.stop_lines.append(stop_line_pose)
                self.stop_lines_closest_wp.append(self.get_closest_waypoint(stop_line_pose))

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        if self.initialized:
            light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                if self.last_state != self.state:
                    rospy.loginfo('Current light state: %d', state)
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    def get_2D_euc_dist(self, pos1, pos2):
        return math.sqrt((pos1.x-pos2.x)**2 + (pos1.y-pos2.y)**2)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        distances = []
        pos = pose.position
        for wp in self.waypoints:
            wp_pos = wp.pose.pose.position
            distances.append(self.get_2D_euc_dist(wp_pos, pos))
        return distances.index(min(distances))

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world
        Returns:
            bbox_topleft (int, int): x, y coordinate of bounding box cutting out traffic light in image
            bbox_bottomright (int, int): x, y coordinate of bounding box cutting out traffic light in image
        """

        # Focal length manually tweaked values from Udacity forum discussion:
        # https://discussions.udacity.com/t/focal-length-wrong/358568/25
        fx = 2574
        fy = 2744
        # Get focal lengths from config file alternatively:
        #TODO: Udacity is going to provide correct values for the focal length
        # fx = self.config['camera_info']['focal_length_x']
        # fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # Get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                                           "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                                                         "/world", now)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        # Use tranform and rotation to calculate 2D position of light in image
        if (trans != None):
            # Convert rotation vector from quaternion to euler:
            euler = tf.transformations.euler_from_quaternion(rot)
            sinyaw = math.sin(euler[2])
            cosyaw = math.cos(euler[2])

            # Rotation followed by translation
            px = point_in_world.x
            py = point_in_world.y
            pz = point_in_world.z
            xt = trans[0]
            yt = trans[1]
            zt = trans[2]
            # Original equations on translation and rotation:
            # Source: http://planning.cs.uiuc.edu/node99.html
            Rnt = (
                px * cosyaw - py * sinyaw + xt,
                px * sinyaw + py * cosyaw + yt,
                pz + zt)

            # Perspective transformation based on Pinhole camera model w/o distortion
            # Source: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
            # Original equations:
            # u = int(fx * -Rnt[1]/Rnt[0] + image_width/2)
            # v = int(fy * -Rnt[2]/Rnt[0] + image_height/2)
            # Tweaked equations:
            u = int(fx * -Rnt[1] / Rnt[0] + image_width / 2 - 30)
            v = int(fy * -(Rnt[2] - 1.0) / Rnt[0] + image_height + 50)

            # Get bounding boxes in order to cut out traffic lights
            # Traffic light's true size
            width_true = 1.0
            height_true = 1.95
            # Get distance traffic light to camera/car
            distance = self.get_2D_euc_dist(self.pose.pose.position, point_in_world)
            # Get size of traffic light within 2D picture
            width_apparent = 2 * fx * math.atan(width_true / (2 * distance))
            height_apparent = 2 * fx * math.atan(height_true / (2 * distance))
            # Get points for traffic light's bounding box
            bbox_topleft = (int(u - width_apparent / 2), int(v - height_apparent / 2))
            bbox_bottomright = (int(u + width_apparent / 2), int(v + height_apparent / 2))
        else:
            bbox_topleft = (0, 0)
            bbox_bottomright = (0, 0)
        return (bbox_topleft, bbox_bottomright)

    def image_resize(self, scr_img, des_width, des_height):
        """Resizes an image while keeping aspect ratio
        Args:
            scr_img: image input to resize
            des_width: pixel width of output image
            des_height: pixel height of output image
        Returns:
            Image: Resized image
        """
        #aspect_ratio_width = des_width / des_height
        # Set manually to 0.5 because division 30/60 apparently results in 0
        aspect_ratio_width = 0.5
        aspect_ratio_height = des_height/des_width
        src_height, src_width = scr_img.shape[:2]
        crop_height = int(src_width / aspect_ratio_width)
        height_surplus = (src_height - crop_height) / 2
        crop_width = int(src_height / aspect_ratio_height)
        width_surplus = (src_width - crop_width) / 2
        # Crop image to keep aspect ratio
        if height_surplus > 0:
            crop_img = scr_img[int(height_surplus):int(src_height-math.ceil(height_surplus)), 0:int(src_width)]
        elif width_surplus > 0:
            crop_img = scr_img[0:int(src_height), int(width_surplus):int(src_width-math.ceil(width_surplus))]
        else:
            crop_img = scr_img
        # Resize image
        return cv2.resize(crop_img, (des_width, des_height), 0, 0, interpolation=cv2.INTER_AREA)
    
    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # Decide for which method is used to send images to classifier
        extract_traffic_light = False

        # Approach no. 1
        # Extract traffic light from camera image
        if extract_traffic_light:
            # Convert given traffic light coordinates into position within 2D image
            bbox_topleft, bbox_bottomright = self.project_to_image_plane(light.pose.pose.position)
            # Use bounding box to extract traffic light
            tl_image_extracted = cv_image[bbox_topleft[1]:bbox_bottomright[1], bbox_topleft[0]:bbox_bottomright[0]]
            # Resize image for classifier to match with training data
            tl_image = self.image_resize(tl_image_extracted, 30, 60)
            # Forward traffic light image to classifier to get prediction
            tl_state = self.light_classifier.get_classification(tl_image)

        # Approach no. 2
        # Use whole camera image
        else:
            # Forward entire image captured by camera to classifier to get prediction
            tl_state = self.light_classifier.get_classification(cv_image)

        rospy.logdebug("status of traffic light: %i" % tl_state)
        return tl_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        VISIBLE_THRESHOLD = 70
        light = None
        idx_next_light = -1

        if self.pose and self.stop_lines:
            car_wp = self.get_closest_waypoint(self.pose.pose)
            bigger_wp = [wp for wp in self.stop_lines_closest_wp 
                              if wp > car_wp]
            if len(bigger_wp) > 0:
                idx_next_light = self.stop_lines_closest_wp.index(min(bigger_wp))
            else:
                idx_next_light = 0
            next_stop_pos = self.stop_lines[idx_next_light].position
            dist_to_next_stop = self.get_2D_euc_dist(self.pose.pose.position, next_stop_pos)
            if dist_to_next_stop <= VISIBLE_THRESHOLD:
                light = self.lights[idx_next_light]

        if light:
            state = self.get_light_state(light)
            return self.stop_lines_closest_wp[idx_next_light], state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

