#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
beef = "hello"

def callback(action):

    beef = action
    print(action)
    return 1

def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("Action_Libary", String, callback)
    print(beef)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
