import sys
import os
import time

# Add the parent directory of 'LeArm' to the Python path
# This is crucial so that Python can find 'LeArm.ik' and 'LeArm.Controller.LeArm'
# Assuming main.py is in 'src/' and LeArm directory is sibling to 'src/'
# Project Root (e.g., robo_chess)
# ├── LeArm/
# │   ├── Controller/
# │   │   └── LeArm.py
# │   ├── Kinematics/
# │   │   ├── arm_attributes.yaml
# │   │   └── matrix.py
# │   └── ik.py
# └── src/
#     └── main.py

# Get the directory of the current script (main.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from 'src')
project_root_dir = os.path.dirname(current_script_dir)
# Add the 'LeArm' directory to sys.path
learm_dir = os.path.join(project_root_dir, 'LeArm')
if learm_dir not in sys.path:
    sys.path.insert(0, learm_dir)
    print(f"Added {learm_dir} to sys.path")

# Now you can import from LeArm.ik and LeArm.Controller
try:
    from ik import InverseKinematics
    from Controller.LeArm import LeArm, LSC_Series_Servo
    print("Successfully imported InverseKinematics and LeArm modules.")
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Check your PYTHONPATH and file structure. Error: {e}")
    print("Expected structure: project_root/LeArm/ik.py, project_root/LeArm/Controller/LeArm.py")
    sys.exit(1)


def main():
    """
    Main function to initialize the LeArm, use Inverse Kinematics to calculate
    joint angles for a target position, and move the arm.
    """
    
    print("Initializing Inverse Kinematics solver...")
    try:
        ik_solver = InverseKinematics()
        print("Inverse Kinematics solver initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize InverseKinematics solver. Ensure arm_attributes.yaml is correct and accessible. Error: {e}")
        sys.exit(1)

    print("Initializing LeArm controller...")
    arm = None
    try:
        # Set debug=True to see HID communication messages from LeArm.py
        arm = LeArm(debug=True)
        print("LeArm controller initialized successfully.")

        # It's good practice to move the arm to a known, safe initial position
        # before attempting IK moves, as the arm doesn't report its state.
        # These angles should be safe for your arm.
        initial_safe_angles = [90, 90, 90, 90, 90, 90] # Example: mid-range for all servos
        print(f"Moving arm to initial safe position: {initial_safe_angles} degrees.")
        arm.servoMove([arm.servo1, arm.servo2, arm.servo3, arm.servo4, arm.servo5, arm.servo6], initial_safe_angles, time=1500)
        time.sleep(1.5) # Give the arm time to reach the position

    except Exception as e:
        print(f"WARNING: Could not initialize LeArm controller. Physical arm movement will be skipped. Error: {e}")
        print("Please ensure the LeArm is connected and the necessary drivers/permissions are set up.")
        # If arm initialization fails, we can still run IK calculations, just not move the physical arm.
        arm = None 
        # Use initial_safe_angles for IK calculations even if physical arm is not connected
        initial_safe_angles = [90, 90, 90, 90, 90, 90] # Fallback for IK starting point


    # --- Define your desired target position in 3D space ---
    # These coordinates are in the same units as your linkVectors in arm_attributes.yaml
    # (e.g., cm or mm). Make sure they are reachable by your arm.
    target_x = 15.0  # Example: 15 units forward from the base
    target_y = 5.0   # Example: 5 units to the side
    target_z = 20.0  # Example: 20 units above the base

    print(f"\nAttempting to move LeArm to target position: (X={target_x}, Y={target_y}, Z={target_z})")

    # Solve Inverse Kinematics to get the required joint angles
    # Pass the current (or initial safe) angles as a starting point for the solver
    solved_servo_angles_deg = ik_solver.solve_ik(target_x, target_y, target_z, current_servo_angles_deg=initial_safe_angles)

    if solved_servo_angles_deg:
        print(f"IK solution found (servo angles in degrees): {np.array(solved_servo_angles_deg).round(2)}")
        
        if arm:
            # Prepare the list of servo objects to control
            servos_to_control = [arm.servo1, arm.servo2, arm.servo3, arm.servo4, arm.servo5, arm.servo6]
            
            # Ensure the number of solved angles matches the number of servos
            if len(solved_servo_angles_deg) == len(servos_to_control):
                print(f"Commanding LeArm to move to the calculated position...")
                # The 'time' parameter specifies the duration of the movement in milliseconds
                arm.servoMove(servos_to_control, solved_servo_angles_deg, time=2000)
                print("LeArm move command sent. Waiting for movement to complete...")
                time.sleep(2.5) # Wait for the arm to complete its movement
                print("Movement complete.")
            else:
                print("ERROR: Mismatch between the number of solved angles and the number of physical servos.")
        else:
            print("Physical LeArm not connected or initialized. Skipping arm movement.")
    else:
        print("Failed to find a valid Inverse Kinematics solution for the target position.")

    # Optional: Get battery voltage after movements
    if arm:
        try:
            voltage = arm.getBatteryVoltage()
            print(f"\nLeArm Battery Voltage: {voltage} V")
        except TypeError as e:
            print(f"Could not read battery voltage: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while getting battery voltage: {e}")
        finally:
            # It's good practice to unload servos to save power and prevent overheating
            print("Unloading servos...")
            arm.servoUnload([arm.servo1, arm.servo2, arm.servo3, arm.servo4, arm.servo5, arm.servo6])
            print("Servos unloaded.")

if __name__ == "__main__":
    main()
