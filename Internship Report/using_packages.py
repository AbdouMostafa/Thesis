#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from FX_ROS import *
import rospy
#from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from niryo_robot_msgs.msg import RobotState

from niryo_robot_msgs.srv import SetBool, SetBoolRequest, SetInt, SetIntRequest, Trigger
from tools_interface.srv import ToolCommand, ToolCommandRequest
from niryo_robot_arm_commander.srv import GetFK, GetFKRequest

import csv
#import moveit_commander

rospy.init_node('controller1')

#robot = moveit_commander.RobotCommander()
#scene = moveit_commander.PlanningSceneInterface()

def req_cal():
    req_cal_service = '/niryo_robot/joints_interface/request_new_calibration'
    Call_Aservice(req_cal_service, type=Trigger, request_name=None)

def learning_mode(on_or_off=False):
    """The function takes argument False or or True, whereas, True = learning more is turned on, 
	and False = learning mode is off."""
    Call_Aservice('/niryo_robot/learning_mode/activate', SetBool, SetBoolRequest, req_args={'value':on_or_off})

def motor_cal():
    cal_service = '/niryo_robot/joints_interface/calibrate_motors'
    Call_Aservice(cal_service, type=SetInt, request_name=SetIntRequest, req_args={"value":1})

def open_gripper():
    open_gripper_service = '/niryo_robot/tools/open_gripper'
    open_gripper_args = {'id':11, 'position':2060, 'speed':500, 
                     'hold_torque': 50, 'max_torque': 100}
    Call_Aservice(open_gripper_service, ToolCommand, ToolCommandRequest, open_gripper_args)

def close_gripper():
    close_gripper_service = '/niryo_robot/tools/close_gripper'
    close_gripper_args = {'id':11, 'position':807, 'speed':500, 
                     'hold_torque': 100, 'max_torque': 100}
    Call_Aservice(close_gripper_service, ToolCommand, ToolCommandRequest, close_gripper_args)

def pick_place_pose(pick_pose, place_pose):
    """"""
    pick_pose0 = list(pick_pose)
    pick_pose0[2] += 0.09

    place_pose0 = list(place_pose)
    place_pose0[2] += 0.09

    set_speed(1)
    Move_to_pose(pick_pose0)

    wait(1)
    set_speed(0.4)
    Move_to_pose(pick_pose)

    wait(0.5)
    close_gripper()

    Move_to_pose(pick_pose0)

    set_speed(1)
    Move_to_pose(place_pose0)

    wait(1)
    set_speed(0.4)
    Move_to_pose(place_pose)

    wait(0.5)
    open_gripper()

    Move_to_pose(place_pose0)

    set_speed(1)
    arm.set_named_target("resting")
    arm.execute(arm.plan(),wait=True)

    print("The pick & place process is being done successfully!")

pick_pose = (0.089, -0.226, 0.053, -0.0835601527521, 0.0279702172245, -1.65055925368)
place_pose = (0.186, -0.232, 0.052, -0.10394094621, -0.025034628265, -1.60465645902)

#pick_place_pose(pick_pose, place_pose)
#Move_to_pose(place)

def pick_place_joints(pick_joints, place_joints):
    fk_pick = FK_Moveit(pick_joints)

    fk_pick.position.z += 0.09

    fk_place = FK_Moveit(place_joints)
    fk_place.position.z += 0.09

    set_speed(1)
    move_pose_orn(fk_pick)

    wait(1)
    set_speed(0.4)
    move_to_joints(pick_joints)

    wait(0.5)
    close_gripper()

    Move_pose_axis('z', add=0.09)

    set_speed(1)
    move_pose_orn(fk_place)

    wait(1)
    set_speed(0.4)
    move_to_joints(place_joints)

    wait(0.5)
    open_gripper()

    Move_pose_axis('z', add=0.09)

    set_speed(1)
    arm.set_named_target("resting")
    arm.execute(arm.plan(),wait=True)

pick_joints = [-1.333046951392955, -1.000387375547136, -0.6400950446822421, -0.12722775180471535, 1.6658104820540134, -0.02138307744060608]
place_joints = [-0.7653681402062875, -1.0170517792451776, -0.3492254528618751, 0.8085005288055256, 1.4203735559923107, -0.13796561731991464]
print('before')
pick_place_joints(pick_joints, place_joints)
print('after')

def Pouring(take, pour):
    fk_pick = Get_FK(take)

    take0 = [fk_pick.position.x, fk_pick.position.y, fk_pick.position.z, fk_pick.rpy.roll, fk_pick.rpy.pitch, fk_pick.rpy.yaw]
    take0[2] += 0.09

    fk_place = Get_FK(pour)
    pour0 = [fk_place.position.x, fk_place.position.y, fk_place.position.z, fk_place.rpy.roll, fk_place.rpy.pitch, fk_place.rpy.yaw]
    pour0[2] += 0.09

    set_speed(1)
    Move_to_pose(take0)

    wait(1)
    set_speed(0.2)
    move_to_joints(take)

    wait(0.5)
    close_gripper()

    Move_pose_axis('z', add=0.09)

    set_speed(1)
    Move_to_pose(pour0)

    wait(1)
    set_speed(0.1)
    Move_pose_axis('x', add=0.028)
    Move_pose_axis('y', add=0.009)
    Move_pose_axis('z', add=-0.03)
    Move_pose_axis('roll', add=2)

    wait(2)
    Move_pose_axis('roll', add=-2)

    set_speed(1)
    Move_to_pose(take0)

    set_speed(0.2)
    move_to_joints(take)
    open_gripper()

    set_speed(0.4)
    Move_pose_axis('z', add=0.09)

    wait(1)
    set_speed(1)
    arm.set_named_target("resting")
    arm.execute(arm.plan(),wait=True)

"""
#motor_cal()
saved_joints = []

with open('joints_values', 'r') as joints_values:
    csv_reader = csv.reader(joints_values)
    
    for line in csv_reader:
        for i in range(6):
            line[i] = float(line[i])
        saved_joints.append(line)


#take = [-1.1854200219422666,-0.9822080260583629,-0.6370651531007799,0.3774519274096604,1.5875774618718457,-0.045926770046776255]
#pour = [-0.7836312654991562,-1.0322012371524885,-0.3840692060486899,0.7686170283204992,1.4433832678105953,-0.20085882962322632]

#take = saved_joints[0]
#pour = saved_joints[1]

#Pouring(take, pour)

#pose = get_pose()

for i in range(5):
    Move_to_pose([0.29537095654868956, 4.675568598554573e-05, 0.4286678926923855, 0.0017192879795506913, 0.0014037282477544944, 0.00016120358136762693])
    wait(1)
    Move_to_pose([0.16628282674524533, -0.3558624564558515, 0.24121429785777998, -2.0083491136896874, 1.542129937691708, -3.106432543002691])
    wait(1)


fk_pick = Get_FK(take)

take0 = [fk_pick.position.x, fk_pick.position.y, fk_pick.position.z, fk_pick.rpy.roll, fk_pick.rpy.pitch, fk_pick.rpy.yaw]
take0[2] += 0.09

fk_place = Get_FK(pour)
pour0 = [fk_place.position.x, fk_place.position.y, fk_place.position.z, fk_place.rpy.roll, fk_place.rpy.pitch, fk_place.rpy.yaw]
pour0[2] += 0.09


print(pour0)


#wait(1)
#arm.set_named_target("resting")
#arm.execute(arm.plan(),wait=True)
current = Get_FK(joints=Get_joitns()).position
#print(current)
count = 0
Errors = {'x': [], 'y': [], 'z': []}
check1 = 1
check2 = 1
Move_to_pose(pour0)
wait(2)
current = Get_FK(joints=Get_joitns()).position
while (abs(current.x - pour0[0]) >= 0.001) or (abs(current.y - pour0[1]) >= 0.001) or (abs(current.z - pour0[2]) >= 0.002):
    if count >= 20:
        break
    
    Errors['x'].append(abs(current.x - pour0[0]))
    Errors['y'].append(abs(current.y - pour0[1]))
    Errors['z'].append(abs(current.z - pour0[2]))

    count += 1

    if (abs(current.x - pour0[0]) > 0.001) and check1:
        Move_pose_axis('x', add=(pour0[0] - current.x))
        print('(pour0[0] - current.x): {}'.format((pour0[0] - current.x)))
    
    elif (abs(current.y - pour0[1]) > 0.001) and check2:
        print('(pour0[1] - current.y): {}'.format((pour0[1] - current.y)))
        Move_pose_axis('y', add=(pour0[1] - current.y))
        check1 = 0
    
    elif (abs(current.z - pour0[2]) > 0.001):
        print('(pour0[2] - current.z): {}'.format((pour0[2] - current.z)))
        Move_pose_axis('z', add=(pour0[2] - current.z))
        check2 = 0

    current = Get_FK(joints=Get_joitns()).position

    print('Trial number number {} and the current values are: \n{}'.format(count, current))
    if (abs(current.x - pour0[0]) <= 0.001) and (abs(current.y - pour0[1]) <= 0.001) and (abs(current.z - pour0[2]) <= 0.001):
        print("\n\n Done, there are the current values: {}".format(current))
#Move_pose_axis('roll', add=2)

xerros = np.array(Errors['x'])
yerros = np.array(Errors['y'])
zerros = np.array(Errors['z'])

x_axis = np.array(range(1,count+1,1))

plt.plot(x_axis, xerros, label='Error in x_axis')
plt.plot(x_axis, yerros, label='Error in y_axis')
plt.plot(x_axis, zerros, label='Error in z_axis')
plt.legend()
plt.show()
"""