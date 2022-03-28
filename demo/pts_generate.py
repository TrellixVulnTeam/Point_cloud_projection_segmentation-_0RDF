#!/usr/bin/env python
# coding: utf-8
# 不要和r2live抢内存
# In[1]:


import rospy
# import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import time
import sys

# In[2]:


import datetime

datetime.datetime.now()
ISOTIMEFORMAT = '%Y_%m_%d_%H_%M_%S'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)

# In[3]:


global d, n, points
points = []
d = None
n = 0


def callback_pcl(data):
    global d
    d = data
    f = open(theTime + ".pts", 'a')
    for ip in points:
        f.write(str(ip[0]))
        f.write(" ")
        f.write(str(ip[1]))
        f.write(" ")
        f.write(str(ip[2]))
        f.write(" ")
        f.write(str(ip[3]))
        f.write(" ")
        f.write(str(ip[4]))
        f.write(" ")
        f.write(str(ip[5]))
        f.write(" ")
        f.write(str(ip[6]))
        f.write('\n')
    f.close()


def Point_cloud():
    global d, n
    ss = point_cloud2.read_points(d, field_names=("x", "y", "z", "intensity", 'r', 'g', 'b'), skip_nans=True)
    ss = list(ss)
    n = n + len(ss)
    return ss


# In[ ]:


if __name__ == '__main__':
    rospy.init_node('points', anonymous=False)
    rospy.Subscriber('/pointcloud_color', PointCloud2, callback_pcl)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if not d is None:
            points = Point_cloud()


        else:
            print("not points_data")
            time.sleep(2)
        rate.sleep()

# In[ ]:
