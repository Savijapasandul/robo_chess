from Controller.LeArm import LeArm, LSC_Series_Servo

arm = LeArm(debug=True)

arm.servo1.position = 1500
arm.servo2.position = 1500
arm.servo3.position = 1500
arm.servo4.position = 1500
arm.servo5.position = 1500
arm.servo6.position = 1500

servos = [arm.servo6, arm.servo5, arm.servo4, arm.servo3, arm.servo2, arm.servo1]
arm.servoMove(servos, [90, 90, 90, 90, 90, 90], time=2000)