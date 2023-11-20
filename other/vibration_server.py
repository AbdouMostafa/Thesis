#!/usr/bin/env python

import rospy
from fx_powder_dosage.srv import VibrationMotor, VibrationMotorRequest, VibrationMotorResponse
from niryo_robot_rpi.srv import SetDigitalIO, SetDigitalIORequest
import time

def digital_out_do4(value=0):
    """Making a DO4 pin ON or OFF.
    
    param: value (int) -> off for 0, and on for any other value.
    """
    service_name = '/niryo_robot_rpi/set_digital_io'
    rospy.wait_for_service(service_name, 2)

    service_call = rospy.ServiceProxy(service_name, SetDigitalIO)
    request = SetDigitalIORequest()

    request.name = 'DO4'
    request.value = bool(value)

    response = service_call(request)


def callback_fun(req):
    on_off = req.on_off # 1 for ON, 0 for OFF
    freq = req.freq # The frequency [Hz]
    duty_cyc = req.duty_cycle # [%]
    time_on = req.time_on # 0 if you want the motor to run until you shut it off. [s]
    period = 1/freq

    start = time.time()
    end = 0

    if (time_on > 0) and (on_off == 1):
        # we need to make this happening for a certain number of second (time_on)!
        while ((end-start) <= time_on):
            digital_out_do4(1) # send a high singal to DO4
            time.sleep((duty_cyc/100)*period)
            digital_out_do4(0) # send a low singal to DO4
            time.sleep((1 - duty_cyc/100)*period)
            end = time.time()
            
        print("The duration of the vibration is: {}".format(end-start))
        
    elif (time_on == 0) and (on_off == 1):
        digital_out_do4(1)
    else:
        digital_out_do4(0)

    response = VibrationMotorResponse(bool(on_off))
    return response


def motor_server():
    rospy.init_node('vibration_server')
    s = rospy.Service('vibration_motor', VibrationMotor, callback_fun)
    print('Server Is called!')
    rospy.spin()

if __name__ == "__main__":
    motor_server()