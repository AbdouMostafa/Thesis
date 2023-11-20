#!/usr/bin/env python
import tf
import time
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt

import rospy
#from ActionClient import ActionClient
from sensor_msgs.msg import JointState
from niryo_robot_arm_commander.srv import GetFK, GetFKRequest, GetJointLimits
#from niryo_robot_msgs.srv import SetBool, SetBoolRequest, SetInt, SetIntRequest, Trigger

#from niryo_robot_arm_commander.msg import ArmMoveCommand, RobotMoveGoal, RobotMoveAction

from moveit_msgs.srv import GetPositionFK, GetPositionIK

from std_msgs.msg import Header
from moveit_msgs.msg import RobotState as RobotStateMoveIt

from geometry_msgs.msg import Pose
import geometry_msgs
from niryo_robot_msgs.msg import RobotState
import moveit_commander
import moveit_msgs.msg
import actionlib

#robot = moveit_commander.RobotCommander()
#scene = moveit_commander.PlanningSceneInterface()
arm = moveit_commander.move_group.MoveGroupCommander("arm")

global Plan


def Call_Aservice(service_name, type, request_name=None, req_args=None, should_return=None):
    
    """
    Paramters:
    
    ........................
    service_name: str
    
    type: srv
    
    request_name: None (srv)
    
    req_args: None (dictionary) ex. {'positon': 210, 'id': 11, 'value': False} 
    
    should_return ?: None (int) >> is set to 1, if you want to return the reponse of the service.
    

    Returns:

    ........................

    If should_return is set to 1, the function is going to return the response of the service.
    Otherwise, the function should only call the service to do a certain action with no return.
    """
    try:
        rospy.wait_for_service(service_name, 2)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Timeout and the Service was not available : " + str(e))
        return RobotState()
    
    try:
        service_call = rospy.ServiceProxy(service_name, type)

        if request_name == None:
            response = service_call()
        else:
            request = request_name()
            for key, value in req_args.items():
                #print("f{key} = {value}")
                method = setattr(request, key, value)
            response = service_call(request)

    except rospy.ServiceException as e:
        rospy.logerr("Falied to use the Service")
        return RobotState()

    if should_return == 1:
         return response
    
def Subscribe(topic_name, type, msg_args):
    """
    Subscribe to certain topic.

    Paramters:
    
    ........................
    topic_name: str
    
    type: srv
    
    msg_args: list >> list of strings, which contains the aguments that we need to read from the topic.
    

    Returns:

    ........................

    Return a list of the readed values from each argument.
    If we have only one argument, it returns the value of this agument only, not a list.
    """
    #rospy.init_node('FX_ROS_Subscriber')

    msg = rospy.wait_for_message(topic_name, type)
    value = []

    if len(msg_args) == 1:
        value = getattr(msg, msg_args[0])
    else:
        for i in msg_args:
            value.append(getattr(msg, i))

    return value

#joints_values = Subscribe('/joint_states', JointState, ["position"])

def Get_joitns():
    joints_values = Subscribe('/joint_states', JointState, ["position"])

    return joints_values

def get_pose():
    return Subscribe('/niryo_robot/robot_state', RobotState, ['position', 'rpy'])

def get_pose_list():
    pose = get_pose()
    position = pose[0]
    rpy = pose[1]

    return [position.x, position.y, position.z, rpy.roll, rpy.pitch, rpy.yaw]

def Get_FK(joints):
    """
    Give the the joints' values to the forward kinematics service,
    and get the pose coordinations.
    """

    fk_service = '/niryo_robot/kinematics/forward'
    return Call_Aservice(fk_service, GetFK, GetFKRequest, {'joints':joints}, should_return=1).pose

def FK_Moveit(joints):
    """
    Get Forward Kinematics from the MoveIt service directly after giving joints
    :param  joints
    :type   joints: list of joints values
    :return: A Pose state object
    @example of a return

position: 
  x: 0.278076372862
  y: 0.101870353599
  z: 0.425462888681
orientation: 
  x: 0.0257527874589
  y: 0.0122083384395
  z: 0.175399274203
  w: 0.984084775322

    """

    from moveit_msgs.srv import GetPositionFK
    from moveit_msgs.msg import RobotState as RobotStateMoveIt
    from std_msgs.msg import Header

    rospy.wait_for_service('compute_fk', 2)
    moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
    
    fk_link = ['base_link', 'tool_link']
    header = Header(0, rospy.Time.now(), "world")
    rs = RobotStateMoveIt()

    rs.joint_state.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    rs.joint_state.position = joints

    reponse = moveit_fk(header, fk_link, rs)

    return reponse.pose_stamped[1].pose


def Move_pose_axis(axis, new=None, add=None):
    """
    Parameters:
    You should either put a value to add or new, not both.
    ........................

    axis: str -> (x, y, z, roll, pitch, or yaw)

    new: float -> The new coordination you want to give to a certain axis.
    "new" will always overright the value of the axis.

    add: float -> the value in meters or radians you want to add to a certain axis.
    """
    global Plan
    
    #FK = Get_FK(Get_joitns())
    FK = get_pose()
    axises = ['x','y','z']

    pose = Pose()
    p_goal = pose.position
    orn_goal = pose.orientation

    p_current = FK[0]
    
    rpy_current = FK[1]
    
    if add:
        if axis.lower() in axises:
            current_value = getattr(p_current, axis)
            setattr(p_current, axis, current_value+add)
        else:
            current_value = getattr(rpy_current, axis)
            setattr(rpy_current, axis, current_value+add)
    if new:
        if axis.lower() in axises:
            setattr(p_current, axis, new)
        else:
            setattr(rpy_current, axis, new)
    

    p_goal.x = p_current.x
    p_goal.y = p_current.y
    p_goal.z = p_current.z

    orn_goal.x, orn_goal.y, orn_goal.z, orn_goal.w = tf.transformations.quaternion_from_euler(rpy_current.roll,rpy_current.pitch,rpy_current.yaw)

    #arm.set_goal_tolerance(0.01)
    arm.set_pose_target(pose)
    Plan = arm.go(wait=True)
    
    arm.stop()
    arm.clear_pose_targets()


def Move_to_pose(pose_values):
    """
    Parameters:
    ........................

    pose_values: list or tuble -> [x, y, z, roll, pitch, yaw]
    """
    global Plan

    pose = Pose()
    p_goal = pose.position
    orn_goal = pose.orientation

    p_goal.x = pose_values[0]
    p_goal.y = pose_values[1]
    p_goal.z = pose_values[2]

    roll = pose_values[3]
    pitch = pose_values[4]
    yaw  = pose_values[5]

    orn_goal.x, orn_goal.y, orn_goal.z, orn_goal.w = tf.transformations.quaternion_from_euler(roll,pitch,yaw)

    #arm.set_goal_tolerance(0.001)
    arm.set_pose_target(pose)
    Plan = arm.go(wait=True)
    
    arm.stop()
    arm.clear_pose_targets()

def move_to_joints(joints):
    """
    Parameters:
    ........................

    joints: list or tuble -> [joint1, joint2, joint3, joint4, joint5, joint6]
    """
    joints_limits = Get_Joints_limits()

    for i in range(6):
        if joints_limits.joint_limits[i].max < joints[i] or joints[i] < joints_limits.joint_limits[i].min:
            print("The joint{} can not be more than {} neither less than {}".format(i+1, joints_limits.joint_limits[i].max, joints_limits.joint_limits[i].min))
            return
        else:
            pass

    #arm.set_joint_value_target(joints)
    arm.go(joints, wait=True)

    arm.stop()

def Move_joint_axis(axis, new=None, add=None):
    """
    Parameters:
    You should either put a value to add or new, not both.
    ........................

    axis: int -> the number of the joint that you want to move

    new: float -> The new coordination you want to give to a joint (axis).
    "new" will always overright the value of the axis.

    add: float -> the value in meters change in a certain joint (axis).
    """

    moving_joints = list(Get_joitns())

    if new:
        moving_joints[axis-1] = new
    elif add:
        moving_joints[axis-1] += add

    joints_limits = Get_Joints_limits()

    if joints_limits.joint_limits[axis-1].max < moving_joints[axis-1] or moving_joints[axis-1] < joints_limits.joint_limits[axis-1].min:
        print("The joint{} can not be more than {} neither less than {}".format(axis, joints_limits.joint_limits[axis-1].max, joints_limits.joint_limits[axis-1].min))
        return 0
    else:
        pass
    
    arm.set_joint_value_target(moving_joints)
    arm.go(moving_joints, wait=True)

    arm.stop()

def Get_Joints_limits():
    """Getting the limits for each joint 
    You can get any joint limits as following:

    Get_Joints_limits().joint_limits[0 - 5].max (float)
    Get_Joints_limits().joint_limits[0 - 5].min (float)
    Get_Joints_limits().joint_limits[0 - 5].name (str)

    Where 0 for (joint 1), and 5 for (joint 6)
    max, min, or name would give the maximum, minimum or name of the indicated joint"""

    return Call_Aservice('/niryo_robot_arm_commander/get_joints_limit', GetJointLimits, should_return=1)

def set_speed(speed):

    arm.set_max_velocity_scaling_factor(speed)

def wait(duration):
    """
    wait for a certain time.

    :param duration: duration in seconds
    :type duration: float
    :rtype: None
    """
    time.sleep(duration)

def move_with_action(pose):
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('simple_action', anonymous=True)

    robot_arm = moveit_commander.move_group.MoveGroupCommander("arm")

    robot_client = actionlib.SimpleActionClient('execute_trajectory', moveit_msgs.msg.ExecuteTrajectoryAction)
    robot_client.wait_for_server()
    #rospy.loginfo('Execute Trajectory server is available for robot')

    robot_arm.set_pose_target(pose)
    #robot_arm.set_pose_target([0.29537095654868956, 4.675568598554573e-05, 0.4286678926923855, 0.0017192879795506913, 0.0014037282477544944, 0.00016120358136762693])
    robot_plan_home = robot_arm.plan()

    robot_goal = moveit_msgs.msg.ExecuteTrajectoryGoal()
    robot_goal.trajectory = robot_plan_home

    robot_client.send_goal(robot_goal)
    robot_client.wait_for_result()
    robot_arm.stop()

def move_pose_orn(pose):

    arm.set_pose_target(pose)
    Plan = arm.go(wait=True)
    
    arm.stop()
    arm.clear_pose_targets()

#Errors = {'x': [], 'y': [], 'z': []}
#current = get_pose()[0]

def pick_place(forward, goal):

    for i in range(50):

        Move_to_pose(forward)
        wait(0.5)
        #current = get_pose()[0]
        #Errors['x'].append(abs(current.x - forward[0]))
        #Errors['y'].append(abs(current.y - forward[1]))
        #Errors['z'].append(abs(current.z - forward[2]))
        wait(0.5)

        Move_to_pose(goal)
        wait(0.5)
        #current = get_pose()[0]
        #Errors['x'].append(abs(current.x - goal[0]))
        #Errors['y'].append(abs(current.y - goal[1]))
        #Errors['z'].append(abs(current.z - goal[2]))
        wait(0.5)
        print('Trial No. {} is done successfully'.format(i*2))

"""
forward = [0.295342266371, 0.000424181627603, 0.428815481717, 0.0124571575603, -0.00166414368617, 0.00166002633903]
goal = [0.166258547286, -0.355808646681, 0.240860080473, -2.00946532074, 1.54214885795, -3.10597468128]

pick_place(forward, goal)

xerros = np.array(Errors['x'])
yerros = np.array(Errors['y'])
zerros = np.array(Errors['z'])

x_axis = np.array(range(1,len(Errors['x'])+1,1))

plt.plot(x_axis, xerros, label='Error in x_axis')
plt.plot(x_axis, yerros, label='Error in y_axis')
plt.plot(x_axis, zerros, label='Error in z_axis')
plt.legend()
plt.show()
"""
