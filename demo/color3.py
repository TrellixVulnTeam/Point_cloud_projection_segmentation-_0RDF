import os
import sys

print('Current working path',
      os.getcwd())  # please enter "/home/jichao/python_ws/Swin-Transformer-Semantic-Segmentation-main/demo" to run the python file
print('当前 Python 解释器路径：', sys.executable)
parent_path = os.path.dirname(sys.path[0])
print('Import libraries from', parent_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)

import cv2
from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointField, CameraInfo
import numpy as np
import tf
import sensor_msgs.point_cloud2
import message_filters  # to synchronize topic subscription
import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs.point_cloud2 import read_points
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry

cameraModel = PinholeCameraModel()

pub_image = {}
pub_pc = {}
# pub_pose = rospy.Publisher('lidar_pose', Pose, queue_size=10)

isRotMatSet = False
# rotationMatrix_lidar_camera = np.array([[0.00561514, -0.999907, -0.0124428,-0.0171173],
#                  [0.0304767, 0.0126084, -0.999456, -0.0587173],
#                  [0.99952, 0.00523287, 0.0305447, -0.0324206],
#                  [0, 0, 0, 1]])
rotationMatrix_lidar_camera = np.array([[-0.0443173, -0.998888, -0.0160588, 0.0677557],
                                        [0.0297446, 0.0147482, -0.999449, -0.019818],
                                        [0.998575, -0.0447705, 0.0290579, 0.24684],
                                        [0, 0, 0, 1]])
cv_image = []
bridge = {}
image_count = 0


def create_pc_fields():
    fields = []
    fields.append(PointField('x', 0, PointField.FLOAT32, 1))
    fields.append(PointField('y', 4, PointField.FLOAT32, 1))
    fields.append(PointField('z', 8, PointField.FLOAT32, 1))
    fields.append(PointField('intensity', 12, PointField.FLOAT32, 1))
    fields.append(PointField('r', 16, PointField.FLOAT32, 1))
    fields.append(PointField('g', 20, PointField.FLOAT32, 1))
    fields.append(PointField('b', 24, PointField.FLOAT32, 1))
    return fields


def RGBD_callback(image_data, pointCloud_data):
    global cv_image
    global bridge
    global cameraModel
    global isRotMatSet
    global rotationMatrix
    global pub_image
    global pub_pc
    global transformMatrix
    global image_count
    # timestr_image = "%.6f" % image_data.header.stamp.to_sec()
    # print(timestr_image)
    # timestr_point = "%.6f" % pointCloud_data.header.stamp.to_sec()
    # print(timestr_point)
    # print("new frame received.")
    try:
        cv_image = bridge.imgmsg_to_cv2(image_data, "bgr8")
        width, height = cv_image.shape[:2]
        # print "cv_image w h = "+str(width) +", "+ str(height)

        cv2.imwrite('./image/' + str(image_count) + '.png', cv_image)
        cv2.imwrite('demo2.png', cv_image)
        image_count += 1

    except CvBridgeError as e:
        print(e)

    if (isRotMatSet):
        result = inference_segmentor(model, args.img)
        segment_image = show_result_pyplot(model, args.img, result, get_palette(args.palette), display=False)
        cv2.imwrite('./demo2_segmented.png', segment_image)

        cv_temp = []
        # cv_temp = cv_image.copy()
        cv_temp = segment_image.copy()
        width, height = cv_temp.shape[:2]

        new_points = []
        for point in (read_points(pointCloud_data, skip_nans=True)):
            pointXYZ = [point[0], point[1], point[2], 1.0]
            intensity = point[3]
            intensityInt = int(intensity * intensity * intensity)
            transformedPoint = rotationMatrix_lidar_camera.dot(transformMatrix.dot(pointXYZ))
            if transformedPoint[2] < 0:
                continue
            projected_2d_point = cameraModel.project3dToPixel(transformedPoint)
            # projection
            if projected_2d_point[0] >= 10 and projected_2d_point[0] <= height - 10 and projected_2d_point[1] >= 10 and \
                    projected_2d_point[1] <= width - 10:
                cv2.circle(cv_temp, (int(projected_2d_point[0]), int(projected_2d_point[1])), 5,
                           (intensityInt % 255, (intensityInt / 255) % 255, (intensityInt / 255 / 255)), thickness=-1)
                [b, g, r] = segment_image[int(projected_2d_point[1]), int(projected_2d_point[0])]
                intensity = result[0][int(projected_2d_point[1])][int(projected_2d_point[0])]  # used as label of segmentation
                new_points.append([point[0], point[1], point[2], intensity, r, g, b])
        try:
            pub_image.publish(bridge.cv2_to_imgmsg(cv_temp, "bgr8"))
            new_pointCloud = sensor_msgs.point_cloud2.create_cloud(pointCloud_data.header, create_pc_fields(),
                                                                   new_points)
            pub_pc.publish(new_pointCloud)
        except CvBridgeError as e:
            print(e)

    else:
        print('Waiting for pose info from sub_pose')


def poseCallback(data):
    global isRotMatSet
    global rotationMatrix
    global transformMatrix
    pose = data
    # print("lidarToRGB, pose received")
    quaternion = (
        pose.pose.pose.orientation.x,
        pose.pose.pose.orientation.y,
        pose.pose.pose.orientation.z,
        pose.pose.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)

    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    translation = [pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z, -1.0]
    rotationMatrix = tf.transformations.euler_matrix(roll, pitch, yaw)
    transformMatrix = rotationMatrix.transpose()
    transformMatrix[:, 3] = -rotationMatrix.transpose().dot(translation)
    # print(transformMatrix)
    isRotMatSet = True


def cameraCallback(data):
    global cameraModel
    cameraModel.fromCameraInfo(data)


def lidarToRGB():
    global pub_image
    global pub_pc
    global bridge
    rospy.init_node('lidar_to_rgb', anonymous=True)

    bridge = CvBridge()

    sub_pose = rospy.resolve_name('/Odometry')

    # subscribe to camera
    sub_pose = rospy.Subscriber(sub_pose, Odometry, callback=poseCallback, queue_size=1)
    camera = rospy.Subscriber(rospy.resolve_name('/camera/color/camera_info'), CameraInfo, callback=cameraCallback,
                              queue_size=1)

    pub_image = rospy.Publisher("image_color_with_proj", Image, queue_size=1)
    pub_pc = rospy.Publisher("pointcloud_color", PointCloud2, queue_size=1)

    sub_image = message_filters.Subscriber('/camera/color/image_raw', Image)
    sub_pointcloud = message_filters.Subscriber('/cloud_registered', PointCloud2)

    ts = message_filters.ApproximateTimeSynchronizer([sub_image, sub_pointcloud], 1, 0.05)
    ts.registerCallback(RGBD_callback)

    rospy.spin()


image_file = './demo2.png'
config_file = '../configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py'
checkpoint_file = '../upernet_swin_tiny_patch4_window7_512x512.pth'
parser = ArgumentParser()
parser.add_argument('--img', default=image_file, help='Image file')
parser.add_argument('--config', default=config_file, help='Config file')
parser.add_argument('--checkpoint', default=checkpoint_file, help='Checkpoint file')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument(
    '--palette',
    default='cityscapes',
    help='Color palette used for segmentation map')
args = parser.parse_args()

# build the model from a config file and a checkpoint file
model = init_segmentor(args.config, args.checkpoint, device=args.device)

if __name__ == '__main__':

    try:
        lidarToRGB()
    except rospy.ROSInterruptException:
        pass
