
#!/usr/bin/env python3
#license removed for brevity

'''
Contact: Dadi Hrannar Davidsson, ddavid20@student.aau.dk
         Mark Kallestrup Keller, mkelle19@student.aau.dk
         Magnus Vestergaard Poulsen, mpouls20@student.aau.dk
'''







from ctypes import pointer
import math
from turtle import position
from cv2 import CC_STAT_MAX

import numpy as np
import rospy
from std_msgs.msg import String, Header
from itounge.msg import RAWItongueOut, process
from kinova_msgs.msg import PoseVelocityWithFingerVelocity, SetFingersPositionAction, SetFingersPositionGoal, ArmPoseAction, ArmPoseGoal, KinovaPose
import actionlib
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import time



class JACO(object):
    cur_pose = [0, 0, 0, 0, 0, 0, 0]
    run = 1
    client = actionlib.SimpleActionClient('/j2n6s200_driver/pose_action/tool_pose', ArmPoseAction)
    point = [0.01 ,-0.7, 0.057]
    orientation_q = [0, 0, 0, 0]
    qw_ = 0
    inputdata = 0
    mode = 0
    open = 0
    picture = 0
    orientation = 0
    camera = process()
    preset = 0
    camera_run = 1
    position_run = 1


    #make the subscribers and the upload rate
    def __init__(self):
        self.action = PoseVelocityWithFingerVelocity()
            
        rospy.Subscriber("RAWItongueOut", RAWItongueOut, self.callback)
        rospy.Subscriber("/j2n6s200_driver/out/cartesian_command", KinovaPose, self.second_callback)
        rospy.Subscriber("camera", process, self.third_callback)
        pub = rospy.Publisher("/j2n6s200_driver/in/cartesian_velocity_with_finger_velocity", PoseVelocityWithFingerVelocity, queue_size=10)
            
        r = rospy.Rate(100)

        while not rospy.is_shutdown():
                
            pub.publish(self.action)
            r.sleep()
    
    #getting the camera coordiantes
    def third_callback(self, msg):
        print("points recived")
        #print(msg.a, msg.b, msg.c)
        JACO.camera.a = msg.a
        JACO.camera.b = msg.b
        JACO.camera.c = msg.c


    #making the camera coordinates fit the robots coordinates and return coordiantes
    def movement(camx, camy, camz, posex, posey, posez):
        #print(camx, camy, camz)
        #print(posez)
        if JACO.orientation == 1:
        


            camy = -camy
            camz = -camz
            x = ((posex*1000)+camz +80)/1000
            y = ((posey*1000)+camx)/1000
            z = ((posez*1000)+camy + 45)/1000
            #print(z)
            if z < 0.050:
                z = 0.050
            cords = [x, y, z]

            return cords
        if JACO.orientation == 2:
            camx = -camx
            camy = -camy
            camz = -camz
            
            x = ((posex * 1000)+camx)/1000
            y = ((posey * 1000)+camz +80)/1000
            z = ((posez * 1000)+camy +45)/1000
            print(z)
            
            if z < 0.050:
                z = 0.050
                print(camy)

            cords = [x, y, z]
            return cords

        if JACO.orientation == 3: 
            camx = -camx
            camy = -camy

            x = ((posex*1000)+camz -80)/1000
            y = ((posey*1000)+camx)/1000
            z = ((posez*1000)+camy +45)/1000
            if z < 0.050:
                z = 0.050
            cords = [x, y, z]
            return cords

    #making the quaternions into eulers
    def QuaternionNorm(Q_raw):
        qx_temp,qy_temp,qz_temp,qw_temp = Q_raw[0:4]
        qnorm = math.sqrt(qx_temp*qx_temp + qy_temp*qy_temp + qz_temp*qz_temp + qw_temp*qw_temp)
        qx_ = qx_temp/qnorm
        qy_ = qy_temp/qnorm
        qz_ = qz_temp/qnorm
        qw_ = qw_temp/qnorm
        JACO.qw_ = qw_
        Q_normed_ = [qx_, qy_, qz_, qw_]
        return Q_normed_

    def Quaternion2EulerXYZ(Q_raw):
        Q_normed = JACO.QuaternionNorm(Q_raw)
        qx_ = Q_normed[0]
        qy_ = Q_normed[1]
        qz_ = Q_normed[2]
        qw_ = Q_normed[3]

        tx_ = math.atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_))
        ty_ = math.asin(2 * qw_ * qy_ + 2 * qx_ * qz_)
        tz_ = math.atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_))
        EulerXYZ_ = [tx_,ty_,tz_]
        return EulerXYZ_

    #making the gripper work - posistion control
    def gripper_client(finger_posistion):

        client = actionlib.SimpleActionClient('/j2n6s200_driver/fingers_action/finger_positions', SetFingersPositionAction)
        client.wait_for_server()

        goal = SetFingersPositionGoal()
        goal.fingers.finger1 = float(finger_posistion[0])
        goal.fingers.finger2 = float(finger_posistion[1])
        goal.fingers.finger3 = 0.0
        client.send_goal(goal)
        if client.wait_for_result(rospy.Duration(5.0)):
            return client.get_result()
        else:
            client.cancel_all_goals()
            rospy.logwarn('        the gripper action timed-out')
            return None

    #chaning the euler back to quaternions
    def EulerXYZ2Quaternion(EulerXYZ_):
        tx_, ty_, tz_ = EulerXYZ_[0:3]
        sx = math.sin(0.5 * tx_)
        cx = math.cos(0.5 * tx_)
        sy = math.sin(0.5 * ty_)
        cy = math.cos(0.5 * ty_)
        sz = math.sin(0.5 * tz_)
        cz = math.cos(0.5 * tz_)

        qx_ = sx * cy * cz + cx * sy * sz
        qy_ = -sx * cy * sz + cx * sy * cz
        qz_ = sx * sy * cz + cx * cy * sz
        qw_ = -sx * sy * sz + cx * cy * cz

        Q_ = [qx_, qy_, qz_, qw_]
        return Q_
    
    #next two are to calculate the the enrty point
    def angulator():
        side_c = math.sqrt(pow(JACO.point[0],2)+ pow(JACO.point[1],2))
        angle = math.asin(JACO.point[0]/side_c)
        angle = angle * 180/math.pi
        #print(f"y angle {angle}")
        return angle

    def entrypoint():
        #print(JACO.point)
        if JACO.point[0] < 0:
            cord = JACO.getPointCoordinatesOnLine([[0 , 0], [JACO.point[0], JACO.point[1]]], 0.15)
            cord = np.asarray(cord)
            entry = [cord[0], cord[1], JACO.point[2]]
            return entry
        if JACO.point[0] > 0:
            cord = JACO.getPointCoordinatesOnLine([[0 , 0], [-JACO.point[0], JACO.point[1]]], 0.15)
            cord = np.asarray(cord)
            entry = [-cord[0], cord[1], JACO.point[2]]
            return entry


    #making the arm stop if it has been working for more than a minute

    def cartesian_pose_client(point, orientation_deg):
        
        orientation_rad = list(map(math.radians, orientation_deg))
        orientation_q = JACO.EulerXYZ2Quaternion(orientation_rad)
        JACO.orientation_q = orientation_q
        JACO.client.wait_for_server()
        goal = ArmPoseGoal()
        goal.pose.header = Header(frame_id=('j2n6s200_link_base'))
        goal.pose.pose.position = Point( x=point[0], y = point[1] , z = point[2])
        goal.pose.pose.orientation = Quaternion( x=orientation_q[0], y=orientation_q[1], z=orientation_q[2], w=orientation_q[3])
        JACO.client.send_goal(goal)

        if JACO.client.wait_for_result(rospy.Duration(60.0)):
            print("done")
            return JACO.client.get_result()
            
        else:
            JACO.client.cancel_all_goals()
            print('        the cartesian action timed-out')
            return None

    

    #making set position presets
    def presets(preset):
        if preset ==  "Home":
            JACO.cartesian_pose_client([-0.4, -0.2, 0.1], [90, JACO.angulator(), 0, 1])

        if preset == "drink":
            JACO.cartesian_pose_client([-0.53367507457733154, -0.1328495442867279, 0.2721599280834198 ], [90, -89.9999, 0, 1])
            JACO.cartesian_pose_client([-0.43367507457733154, 0.1328495442867279, 0.2721599280834198 ], [90, -89.9999, 0, 1])

        if preset == "put_back":
            JACO.cartesian_pose_client([JACO.point[0], JACO.point[1], JACO.point[2] + 0.05], [90, JACO.angulator(), 0, 1])
            #JACO.cartesian_pose_client([JACO.point[0], JACO.point[1], JACO.point[2]], [90, JACO.angulator(), 0, 1])


    #so we can set coordiantes
    def second_callback(self, info):

        JACO.cur_pose = [info.X, info.Y, info.Z, info.ThetaX, info.ThetaY, info.ThetaZ]  
    
    #controls
    def callback(self, data):
        
        

        inputdata = data.Sensor
        if inputdata == 10 and JACO.mode == 0 and JACO.run == 1:
            JACO.mode = 1
            JACO.run = 0
        elif inputdata == 10 and JACO.mode == 1 and JACO.run == 1:
            JACO.mode = 0
            JACO.run = 0

        if JACO.mode == 0:
            JACO.XYZmovement(inputdata)
        elif JACO.mode == 1:
            JACO.XYZangular(inputdata)

        if inputdata == 7 and JACO.open == 0 and JACO.run == 1:
            JACO.gripper_client([6800, 6800])
            JACO.run = 0
            JACO.open = 1
        if inputdata == 7 and JACO.open == 1 and JACO.run == 1:
            JACO.gripper_client([0, 0])
            JACO.run = 0
            JACO.open = 0
        
        if inputdata == 4 and JACO.run == 1:
            JACO.camera_run += 1
            if JACO.camera_run == 40:
                JACO.picture = 1
                #print(JACO.picture)
                JACO.run = 0
                JACO.camera_run = 1
                print("yea yea i get it")
        
        if inputdata == 5 and JACO.run == 1:
            JACO.position_run += 1
            if JACO.position_run == 40:
                JACO.preset = 1
                JACO.run = 0
                JACO.position_run = 1
                print("Where should i go?")


        if JACO.preset == 1 and JACO.run == 1:
            if inputdata == 1:
                JACO.presets("Home")
                JACO.preset = 0
            if inputdata == 2:
                JACO.presets("drink")
                JACO.run = 0
                JACO.preset = 0
            if inputdata == 3:
                JACO.presets("put_back")
                JACO.preset = 0



        if inputdata == 6:
            #print(JACO.camera.a , JACO.camera.c, JACO.cur_pose[0], JACO.cur_pose[1], JACO.cur_pose[2])
            JACO.point =  JACO.movement(JACO.camera.a, JACO.camera.b , JACO.camera.c , JACO.cur_pose[0], JACO.cur_pose[1], JACO.cur_pose[2])
            
            print("I have an object")
            #print(JACO.point)
        if JACO.picture == 1:
            print("Which way shall I turn?")
            if inputdata == 1 and JACO.run == 1:
                JACO.TakePicture(-89)
                JACO.run = 0
                JACO.picture = 0
                JACO.orientation = 1
            elif inputdata == 2  and JACO.run == 1:
                JACO.TakePicture(0)
                JACO.run = 0
                JACO.picture = 0
                JACO.orientation = 2
            elif inputdata == 3  and JACO.run == 1:
                JACO.TakePicture(89)
                JACO.run = 0
                JACO.picture = 0 
                JACO.orientation = 3   
        elif JACO.picture == 0:
            if inputdata == 1 and JACO.run != 0:
                JACO.run += 1
                #confirmation button at a diffirnet point
                if JACO.run == 20:
                    #print(f"entry point {JACO.entrypoint()}")
                    #print(f"end point {JACO.point}")
                    JACO.gripper_client([0.0 , 0.0])
                    JACO.cartesian_pose_client(JACO.entrypoint(), [90, JACO.angulator(), 0, 1])
                    JACO.cartesian_pose_client(JACO.point, [90, JACO.angulator() , 0, 1])
                    JACO.gripper_client([6800.0 , 6800.0])
                    JACO.cartesian_pose_client([JACO.point[0], JACO.point[1], JACO.point[2] + 0.05], [90, JACO.angulator(), 0, 1])
                    JACO.run = 0

          
        
        if inputdata == 500:
            JACO.run = 1
            JACO.camera_run = 1
            JACO.position_run = 1
     
            
    def XYZmovement(input):

        action = PoseVelocityWithFingerVelocity

        if input == 17:
            action.twist_linear_y = -2.0
        elif input == 12:
            action.twist_linear_y = 2.0
        elif input == 15:
            action.twist_linear_x = -2.0    
        elif input == 14:
            action.twist_linear_x = 2.0
        elif input == 18:
            action.twist_linear_y = -2.0
            action.twist_linear_x = -2.0
        elif input == 16:
            action.twist_linear_y = -2.0
            action.twist_linear_x = 2.0
        elif input == 13:
            action.twist_linear_y = 2.0
            action.twist_linear_x = -2.0
        elif input == 11:
            action.twist_linear_y = 2.0
            action.twist_linear_x = 2.0
        elif input == 8:
            action.twist_linear_z = 2.0
        elif input == 9:
            action.twist_linear_z = -2.0
        elif input == 500:
            action.twist_linear_y = 0.0
            action.twist_linear_x = 0.0
            action.twist_linear_z = 0.0
        return action    
        
    def XYZangular(input):
        
        action = PoseVelocityWithFingerVelocity

        if input == 17:
            action.twist_angular_x = 2.0
        elif input == 12:
            action.twist_angular_x = -2.0
        elif input == 15:
            action.twist_angular_y = -2.0    
        elif input == 14:
            action.twist_angular_y = 2.0
        elif input == 18:
            action.twist_angular_y = -2.0
            action.twist_angular_x = -2.0
        elif input == 16:
            action.twist_angular_y = 2.0
            action.twist_angular_x = -2.0
        elif input == 13:
            action.twist_angular_y = -2.0
            action.twist_angular_x = 2.0
        elif input == 11:
            action.twist_angular_y = 2.0
            action.twist_angular_x = 2.0
        elif input == 8:
            action.twist_angular_z = 2.0
        elif input == 9:
            action.twist_angular_z = -2.0
        elif input == 500:
            action.twist_angular_y = 0.0
            action.twist_angular_x = 0.0
            action.twist_angular_z = 0.0
        return action     


    def TakePicture(angle):
        JACO.cartesian_pose_client(JACO.cur_pose[ :3], [90, angle, 0, 1])

    # <editor-fold desc="CALCULATE BLOB TOP and BOTTOM WIDTH RATIO FEATURE">

    # This function returns the new X or Y coordinate on a line determined by two input points
    # and whose location is determined by the given percentage. Whether it is the X or Y coordinate
    # it is determined by which of the X and Y projections resulted from the subtraction of the two points is the largest
    def findLargestMagnitudeOfVectorProjections(vector1, vector2, percentage):
        vector3 = np.absolute(np.subtract(vector1, vector2))
        maxMagnitude = max(vector3[0], vector3[1])
        # If the projection on X is larger
        if maxMagnitude == vector3[0]:
            # Find which point is furthest on X
            maxXVal = max(vector1[0], vector2[0])
            if maxXVal == vector1[0]:
                # Get the X value of the point placed at the chosen percentage with respect
                # to the line between the two vectors length
                newXCoord = percentage * (vector1[0] - vector2[0]) + vector2[0]
            else:
                newXCoord = percentage * (vector2[0] - vector1[0]) + vector1[0]
            xIsLarger = True
            return xIsLarger, newXCoord
        # If the projection on Y is larger
        elif maxMagnitude == vector3[1]:
            # Find which point is furthest on Y
            maxYVal = max(vector1[1], vector2[1])
            if maxYVal == vector1[1]:
                newYCoord = percentage * (vector1[1] - vector2[1]) + vector2[1]
            else:
                newYCoord = percentage * (vector2[1] - vector1[1]) + vector1[1]
            xIsLarger = False
            return xIsLarger, newYCoord


    # Implemented linear interpolation that takes two data points, say (x0,y0) and (x1,y1), and the interpolant
    # is given by the formula presented in the function at the point (x, y). That is if known x or y, we can find the other
    # because of the linearity properties (i.e. 10 % increase in x will have a 10 % increase in y)
    def findYWithXOrXWithYAndInterpolation(isXLarger, y1, y0, x1, x0, largestMagnitude):
        if isXLarger == True:
            # Calculate y given x, here largest magnitude = x
            y = y0 + (y1 - y0) * ((largestMagnitude - x0) / (x1 - x0))
            return y
        else:
            # Calculate x given y, here largest magnitude = y
            x = x0 + (x1 - x0) * ((largestMagnitude - y0) / (y1 - y0))
            return x


    # This function returns the coordinates of a point placed at a specified percentage on a line, given the two
    # points that make up that line
    def getPointCoordinatesOnLine(points, percentage):
        xIsLarger, newCoordBasedOnChosenPercentage = JACO.findLargestMagnitudeOfVectorProjections(points[0], points[1], percentage)
        if xIsLarger == True:
            newYCoord = JACO.findYWithXOrXWithYAndInterpolation(xIsLarger, points[1][1], points[0][1], points[1][0], points[0][0], newCoordBasedOnChosenPercentage)
            return newCoordBasedOnChosenPercentage, newYCoord
        else:
            newXCoord = JACO.findYWithXOrXWithYAndInterpolation(xIsLarger, points[1][1], points[0][1], points[1][0], points[0][0], newCoordBasedOnChosenPercentage)
            return newXCoord, newCoordBasedOnChosenPercentage
    # </editor-fold>

    
if __name__ == '__main__':
    rospy.init_node('Action_library', anonymous=True)
    
    
    try:
        JACO()
        
    except rospy.ROSInterruptException:
           pass

    
        
    


