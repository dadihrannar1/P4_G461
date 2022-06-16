#!/usr/bin/env python3



import rospy

from itounge.msg import RAWItongueOut, sys # you need to change this so it fits your ROS package

import serial
import numpy as np
import time

class TCI(object):
    def __init__(self, COM="/dev/ttyUSB0"):
        self.COM = COM
        self.s = None
        self.ContactThreshold = 0.12
        self.CuttingThreshold = 0.08
        self.InHand = True
        

        self.maxim = np.ones([18, ])*200
        self.minim = np.ones([18, ])*180
        self.Connected = False
        

        self.XY_Coordinates = np.array([0,0])
        self.ActivatedCoil = 0

        self.InMouth = [0,3,2,1,
                        6,5,4,
                        10,9,8,7,
                        13,12,11,
                        15,14,
                        18,17,16]

        self.sensor_positions = self.Sensor_positions()
        self.movingTime = time.time()
        self.sensorsNeighbors =np.array([[1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                         [1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0],
                                         [0,1,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
                                         [0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0],
                                         [0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0],
                                         [0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1],
                                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1]])
        self.sensorsNotNeighbors = 1 - self.sensorsNeighbors

        rospy.init_node('rawitonguetalker', anonymous=True)
        self.pubRAW = rospy.Publisher('RAWItongueOut', RAWItongueOut, queue_size=10)
        self.Connect()

    def NewSysCmd(self, data):
#        self.COM = data.tci_port   #Name of port, should be /dev/ttyUSB0 if you use the USB port and the raw data, you can also comment this out if you don't want to use this, it is initiated at the top
#        self.ContactThreshold = data.ContactThreshold #initial should be 0.12, you can play around with this number, you can also comment this out if you don't want to use this, it is initiated at the top
#        self.CuttingThreshold = data.CuttingThreshold #initial should be 0.08, you can also play around with this number, you can also comment this out if you don't want to use this, it is initiated at the top
#        self.InHand = data.InHand #bool -  if in hand, data is mirrored 
        if data.start_tci: #bool
            self.Connect()
        elif self.Connected:
            self.Connected = False
            print('Disconnected')
        else:
            print('Already disconnected')

    def Connect(self):
#        """ Connect to the serial"""
        try:
            if self.s is None:
                self.s = serial.Serial(self.COM, bytesize=8, timeout=1, stopbits=1, baudrate=115200)
                Sdata = self.s.readline().decode('utf-8')
                if len(Sdata) > 0:
                    print("Successfully connected to port %r." % self.s.port)
                    self.Connected = True
                else:
                    print("Data < 0 => Try again!")
                    self.s = None
            else:
                if self.s.isOpen():
                    self.s.flushInput()
                    self.Connected = True
                    print("Input flushed, re-connected to port %r." % self.s.port)
                else:
                    self.s.open()
                    self.s.flushInput()
                    self.Connected = True
                    print("What dis?")
        except:
            print("No data: Check Serial Connection")
            print("hello")
            self.s = None
            self.Connected = False

    def disconnect(self):
        if self.s is None:
            print("Serial is not yet established")
        elif self.s.isOpen():
            self.s.close()
            print("Disconnected.")
        else:
            self.s.close()
            print("Already closed.")


    def baseLine_check(self, sensors):
        # ###   this function is used to remove the drift of the baseline
        # ### it reset the value for the baseline (self.maxim)

        nonActiveValue = self.sensorsNotNeighbors[sensors.argmin()]*sensors
        self.maxim[np.nonzero(nonActiveValue)] = np.median(sensors)


    def Sensor_positions(self):
        sensor_bitsX = [0.5, 0, -0.5,
                        0.5, 0, -0.5,
                        0.75, 0.25, -0.25, -0.75,
                        0.5, 0, -0.5,
                        0.45, -0.45,
                        0.5, 0, -0.5]
        sensor_bitsY = [1.25, 1.25, 1.25,
                        0.75, 0.75, 0.75,
                        0.25, 0.25, 0.25, 0.25,
                        -0.5, -0.55, -0.5,
                        -1.0, -1.0,
                        -1.5, -1.45, -1.5]


        sensor_positions = np.zeros([18, 2])
        sensor_positions[:, 0] = (np.array(sensor_bitsX)*170.67) # should be between -128 and 128 
        sensor_positions[:, 1] = (np.array(sensor_bitsY)*93.09 + 11.635)
        return sensor_positions

    def ShouldIrun(self):
        return self.Connected

    def ReadXY(self):
        
#        """Read the serial data from the CU, and translate it to XY coordinates"""
        if self.ShouldIrun():

            try:
                Sdata = self.s.readline().decode('utf-8').replace('\x1b[1m\x1b[22m','') #.rstrip()
                if len(Sdata) == 45 and Sdata[0] == 'P':
                    # Get TCI inductance measures:
                    raw = np.zeros([18])
                    for i in range(18):
                        raw[i] = int(Sdata[2*i+4:2*i+6], 16)
                    # update min values:
                    self.minim[raw < self.minim] = raw[raw < self.minim]
  
                    RawActivationPercentage = (self.maxim-raw)/(self.maxim-self.minim)

                    p = [self.CuttingThreshold, 0.98] #Cutting threshold first set to 0.08
 
                    RAWweight = (RawActivationPercentage-p[0])/(p[1]-p[0])
                    RAWweight[RawActivationPercentage<p[0]] *= 0
                    RAWweight[RawActivationPercentage>p[1]] = np.ones(np.sum(RawActivationPercentage>p[1]))

                    # handle activation: 
                    if max(RAWweight)<self.ContactThreshold: 
                        # Not activated
                        if ((time.time() - self.movingTime) < 0.08):
                            self.XY_Coordinates = np.array([500,500])
                            self.ActivatedCoil = 500
                            self.PubParRAWItongue(raw, Sdata)
                            return
                        self.XY_Coordinates = np.array([0,0])
                        self.ActivatedCoil = 0
                        
                        # publish data
                        self.PubParRAWItongue(raw, Sdata) 
                    else:
                        # Activated
                        w = self.sensorsNeighbors[RAWweight.argmax()]*RAWweight
                        self.XY_Coordinates = w.dot(self.sensor_positions)/sum(w) 
                        self.ActivatedCoil = np.argmax(RAWweight)+1 #activated sensor is where the max value is

                        if not self.InHand:
                            self.XY_Coordinates[0] = -self.XY_Coordinates[0]
                            self.ActivatedCoil = self.InMouth[self.ActivatedCoil]

                        self.movingTime = time.time()
               
                        self.baseLine_check(raw) # this function removes the baseline drift
                        # publish data
                        self.PubParRAWItongue(raw, Sdata)

            except:
                rospy.logwarn('TCI lost a package')
                self.XY_Coordinates = np.array([0,0])
                self.ActivatedCoil = 0
                self.PubParRAWItongue(np.zeros([18]), 'None') 


    def PubParRAWItongue(self, Raw, SData):
        data =  RAWItongueOut()
        data.serialdata = SData
        data.raw = Raw
        data.x = self.XY_Coordinates[0]
        data.y = self.XY_Coordinates[1]
        data.Sensor = self.ActivatedCoil
        self.pubRAW.publish(data) 
        print(data.Sensor)
        
    
        
def run():
    tci_handle = TCI()
    rospy.Subscriber('Sys_cmd', sys, tci_handle.NewSysCmd)
    R = rospy.Rate(60) # tci sends 30hz
    while not rospy.is_shutdown():
        tci_handle.ReadXY()

        R.sleep()
        
        

if __name__ == '__main__':
    try:
        
        run()
        
    except rospy.ROSInterruptException:
        tci_handle.disconnect()
