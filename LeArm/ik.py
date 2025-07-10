import numpy as np
import math
import os
import yaml
import random
from scipy.integrate import solve_ivp
import numdifftools as nd

# Assuming 'Controller' and 'Kinematics' are sibling directories or accessible in PYTHONPATH
from Controller.LeArm import LeArm, LSC_Series_Servo
from Kinematics.matrix import matrix # Import your custom matrix class

# Set numpy print options for cleaner output
np.set_printoptions(precision=2, suppress=True)

# --- Functions from your kinematics.py file ---
# These functions are directly integrated here for a self-contained IK file.
# Alternatively, you could import them if kinematics.py is a separate module.

def buildArm():
    '''Load arm attributes from YAML file and construct arm geometry.
    
    Output:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
    '''

    # Adjust path to arm_attributes.yaml based on the common project structure.
    # This assumes ik.py is in 'LeArm/' and arm_attributes.yaml is in 'LeArm/Kinematics/'.
    # Get the directory of the current script (ik.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to arm_attributes.yaml assuming it's in 'Kinematics' folder
    arm_attributes_path = os.path.join(current_dir, "Kinematics", "arm_attributes.yaml")
    
    # Fallback if arm_attributes.yaml is in the same directory as ik.py (less likely for this structure)
    if not os.path.exists(arm_attributes_path):
        arm_attributes_path = os.path.join(current_dir, "arm_attributes.yaml")

    print(f"Attempting to load arm_attributes.yaml from: {arm_attributes_path}")
    with open(arm_attributes_path, "r") as f:
        armData = yaml.safe_load(f)
    
    jointAxis = armData['jointAxis']
    jointAngles = armData['jointAngles']
    linkVectors = []
    for link in armData['linkVectors']:
        linkVectors += [matrix(link)]

    # IMPORTANT: Ensure that the number of entries in jointAngles, jointAxis, and linkVectors
    # in arm_attributes.yaml matches the number of servos you intend to control (e.g., 6 for LeArm).
    # If your arm has 6 servos, but these lists only have 5 entries, you WILL get a dimension mismatch error.
    if not (len(jointAngles) == len(jointAxis) == len(linkVectors)):
        raise ValueError(
            "Mismatch in lengths of jointAngles, jointAxis, or linkVectors in arm_attributes.yaml. "
            "All lists must have the same number of entries (e.g., 6 for a 6-DOF arm)."
        )
    
    return linkVectors, jointAxis, jointAngles

def rotationSet(jointAxis, jointAngles):
    '''Converts a set of jointAxis and jointAngles into a list of rotation matrices.
    
    Input:
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.

    Output:
        R_set: List of rotation matrix objects.
    '''

    # Assign the apropriate rotation matrix to each axis based off its global frame rotation
    R_set = []
    for idx, jointAxis_char in enumerate(jointAxis): # Renamed jointAxis to jointAxis_char to avoid conflict
        cosAlpha = math.cos(jointAngles[idx])
        sinAlpha = math.sin(jointAngles[idx])

        match jointAxis_char:
            case 'x':
                R_set.append(matrix([
                        [1, 0, 0],
                        [0, cosAlpha, -sinAlpha],
                        [0, sinAlpha, cosAlpha]
                        ]))
            case 'y':
                R_set.append(matrix([
                        [cosAlpha, 0, sinAlpha],
                        [0, 1, 0],
                        [-sinAlpha, 0, cosAlpha]
                    ]))
            case 'z':
                R_set.append(matrix([
                        [cosAlpha, -sinAlpha, 0],
                        [sinAlpha, cosAlpha, 0],
                        [0, 0, 1]
                    ]))
            
            case _:
                raise ValueError(f"{jointAxis_char} is not a known joint axis. Use 'x', 'y', or 'z'.")
    
    return R_set

def rotationSetCumulativeProduct(rotationSet):
    '''Compute cumulative rotation matrices.
    
    Input:
        rotationSet: List of rotation matrix objects representing link orientations relative to the previous link.

    Output:
        rotationSetCumulative: List of rotation matrix objects representing link orientations relative to the global frame.
    '''

    # Create a copy to avoid modifying the original list in place
    rotationSetCumulative = list(rotationSet) 

    for idx, rotationMatrix in enumerate(rotationSet):
        # First rotation matrix is already in global coordinates
        if idx == 0:
            continue

        rotationSetCumulative[idx] = rotationSetCumulative[idx - 1] * rotationMatrix

    return rotationSetCumulative

def rotateVectors(vectors, rotationMatrices):
    '''Rotate vectors by rotation matrices.
    
    Input:
        vectors: List of matrix objects representing vectors.
        rotationMatrices: List of matrix objects representing rotation matrices.

    Output:
        rotatedVectors: List of matrix objects representing rotated vectors.
    '''
    rotatedVectors = []
    for i, vector in enumerate(vectors):
        rotatedVectors.append(rotationMatrices[i] * vector)
    
    return rotatedVectors

def vectorSetCumulativeSum(vectors):
    '''Compute cumulative sum of vectors.
    
    Input:
        vectors: List of matrix objects.

    Output:
        vectorsCumulativeSum: List of matrix objects representing cumulative sum of vectors.
    '''

    vectorsCumulativeSum = [vectors[0]]

    for idx, vector in enumerate(vectors):
        # Skip the first entry
        if idx == 0:
            continue

        vectorsCumulativeSum += [vectorsCumulativeSum[idx - 1] + vector]

    return vectorsCumulativeSum

def vectorSetDifference(vector, vectorSet):
    '''Compute difference between a vector and a set of vectors.
    
    Input:
        vector: Matrix object representing a vector.
        vectorSet: List of matrix objects representing vectors.

    Output:
        vectorDifferences: List of matrix objects representing differences between vector and vectors in vectorSet.
    '''

    vectorDifferences = []
    for vector2 in vectorSet:
        vectorDifferences.append(vector - vector2)
    
    return vectorDifferences

def getPosition(linkVectors, jointAxis, jointAngles): # Removed graph and useRigidBodies as they were not used in the original context
    '''Calculate end points of each link in global coordinates.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.

    Output:
        vectorsGlobalFrame: List of matrix objects representing end points of each link in global coordinates.
    '''

    RSet= rotationSet(jointAxis, jointAngles) # Create rotation geometry relative from base link
    rotationSetCumulative = rotationSetCumulativeProduct(RSet) # Create rotation geometry relative to global frame
    vectorsGlobalRotation = rotateVectors(linkVectors, rotationSetCumulative) # Rotate individual vectors to the global frame
    vectorsGlobalFrame = vectorSetCumulativeSum(vectorsGlobalRotation) # Add vectors together so that they each end at the end of their respective link

    return vectorsGlobalFrame

def threeDJointAxisSet(jointAxis):
    '''Generate 3D joint axis vectors.
    
    Input:
        jointAxis: List of characters representing joint rotation axes.

    Output:
        jointAxisVectors: List of matrix objects representing 3D joint axis vectors.
    '''

    jointAxisVectors = []
    for axis in jointAxis:
        match axis:
            case 'x':
                vx = matrix([[1],[0],[0]])
                jointAxisVectors.append(vx)
            case 'y':
                vy = matrix([[0],[1],[0]])
                jointAxisVectors.append(vy)
            case 'z':
                vz = matrix([[0],[0],[1]])
                jointAxisVectors.append(vz)
            case _:
                raise ValueError(f"{axis} is not a known joint axis.")
    
    return jointAxisVectors

def armJacobian(linkVectors, jointAxis, jointAngles, linkNumber):
    '''Compute Jacobian matrix for arm.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        linkNumber: Index of the link.

    Output:
        jacobian: Numpy array representing the Jacobian matrix.
    '''
    # Ensure that the number of joint angles matches the kinematic model's DOF
    if len(jointAngles) != len(linkVectors):
        raise ValueError(
            f"Dimension mismatch: jointAngles ({len(jointAngles)}) "
            f"does not match linkVectors ({len(linkVectors)}) in armJacobian. "
            "Ensure your arm_attributes.yaml defines the correct number of joints."
        )

    linkEnds = getPosition(linkVectors, jointAxis, jointAngles)
    
    vDiff = vectorSetDifference(linkEnds[linkNumber], [matrix([[0], [0], [0]])] + linkEnds)

    jointAxisVectors = threeDJointAxisSet(jointAxis)

    jointRotations = rotationSet(jointAxis, jointAngles)
    worldJointRotations = rotationSetCumulativeProduct(jointRotations)

    rotatedJointAxisVectors = rotateVectors(jointAxisVectors, worldJointRotations)

    jacobian = np.zeros((3, len(linkVectors)))
    for i, vector in enumerate(vDiff):
        # The loop iterates up to len(vDiff) - 1.
        # vDiff has len(linkVectors) + 1 elements.
        # The Jacobian has len(linkVectors) columns.
        # The break condition ensures we don't try to access a column beyond the Jacobian's size.
        if i >= len(linkVectors): # Corrected condition to prevent IndexError
             break

        cross = np.cross(rotatedJointAxisVectors[i].mat.reshape(1, 3), vector.mat.reshape(1, 3))

        jacobian[0][i] = cross[0][0]
        jacobian[1][i] = cross[0][1]
        jacobian[2][i] = cross[0][2]

    return jacobian

def traceShape(linkVectors, jointAxis, shapeFunction, startingJointAngles, T=(0, 1)):
    '''Trace a shape with the arm.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        shapeFunction: Function defining the shape to trace.
        startingJointAngles: List of starting joint angles in radians.
        T: Tuple specifying the time interval, default (0, 1).

    Output:
        alphas: List of joint angles representing the traced shape.
    
    Notes:
        This will trace the shape given by the parametric curve as defined by shapeFunction relative to the arms current end position.
    '''
    
    jacobian = lambda jointAngles_local: armJacobian(linkVectors, jointAxis, jointAngles_local, len(linkVectors) - 1) # Use last link as end-effector
    
    def jointVelocity(t, alpha):
        shapeGradient = nd.Gradient(shapeFunction)
        velocity = shapeGradient(t)
        
        # Ensure alpha (current joint angles) has the correct shape for Jacobian calculation
        # and that the Jacobian's output matches the expected alphaDot dimension.
        # This is where the (5,) vs (6,) mismatch was happening.
        # The Jacobian will now be (3, N) where N is the number of joints from YAML.
        # alphaDot will be (N,).
        
        # Check if the Jacobian is singular or ill-conditioned
        J = jacobian(alpha)
        if np.linalg.det(J @ J.T) < 1e-9: # Check for singularity (J * J_transpose for pseudo-inverse)
            print("WARNING: Jacobian is singular or near-singular. IK solution may be unstable.")
            # You might want to add a small perturbation or return zeros for alphaDot
            # to prevent division by zero or large unstable movements.
            return np.zeros_like(alpha)

        # Use pseudo-inverse for robust solution
        alphaDot = np.linalg.pinv(J) @ velocity

        return alphaDot

    odeResult = solve_ivp(jointVelocity, T, startingJointAngles, dense_output=True)

    evalTimes = np.linspace(0, 1, 100)
    alphas = []
    for time in evalTimes:
        alphas.append(odeResult.sol(time))

    return alphas

def goToPos(linkVectors, jointAxis, jointAngles, desiredEndPos):
    '''Generate joint angles that will move arm to desired position.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        desiredEndPos: Matrix object representing the desired end position.

    Output:
        jointAnglesFinalPos: List of joint angles representing the final position of the arm.
    '''

    if isinstance(desiredEndPos, list):
        desiredEndPos = matrix(desiredEndPos)

    seperationVector = desiredEndPos - getPosition(linkVectors, jointAxis, jointAngles)[-1]

    def goToEndPoint(t):
        if isinstance(t, np.ndarray):
            t=t[0]
        
        x = seperationVector.mat[0][0] * t
        y = seperationVector.mat[1][0] * t
        z = seperationVector.mat[2][0] * t

        v = np.array([x, y, z])

        return v

    jointAnglesFinalPos = traceShape(linkVectors, jointAxis, goToEndPoint, jointAngles)[-1]

    return jointAnglesFinalPos

def moveToPosition(linkVectors, jointAxis, jointAngles, vector, distanceFromPos=0.2, linkNumber=-1, jointLimits=[(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90), (-90, 90)]): # Updated to 6 limits
    '''Generate joint angles that will move arm to desired position. 
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        vector: Matrix object representing a Cartesian vector in the world frame.
        distanceFromPos: Acceptable amount of displacement from requested position, default 0.2.
        linkNumber: Index of the link, default -1.
        jointLimits: List of tuples representing joint limits, default [(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90), (-90, 90)].

    Output:
        jointAngles: List of joint angles representing the arm's position.
    
    Notes:
        I would not recommend using this function, goToPos is faster and more accurate.
    '''

    if np.linalg.norm(vector.mat) > 500:
        raise ValueError(f"The requested position vector {vector} is outside the reach of the arm.")

    def endPointDifferenceMap(jointAngles_local): # Renamed jointAngles to jointAngles_local

        # Force joint limits to be respected in the mapping
        for i, jointLimit in enumerate(jointLimits):
             # Ensure we don't try to access a joint limit that doesn't exist for the current joint index
            if i < len(jointAngles_local):
                if min(jointLimit) > math.degrees(jointAngles_local[i]):
                    jointAngles_local[i] = math.radians(min(jointLimit))
                elif max(jointLimit) < math.degrees(jointAngles_local[i]):
                    jointAngles_local[i] = math.radians(max(jointLimit))
                  
        
        endEffectorPos = getPosition(linkVectors, jointAxis, jointAngles_local)[linkNumber]
        difference = endEffectorPos - vector
        absDifference = np.linalg.norm(difference.mat)

        return absDifference
    
    endPointDifferenceMapGradient = nd.Gradient(endPointDifferenceMap)

    zeroVector = endPointDifferenceMap(jointAngles)

    # Added a safeguard to prevent infinite loops if convergence is not met
    max_iterations = 1000
    iteration = 0

    while zeroVector > distanceFromPos and iteration < max_iterations:
        gradient = endPointDifferenceMapGradient(jointAngles)
        negativeGradient = gradient * -1

        tempJointAngles = []
        for jointAngle, gradientParameter in zip(jointAngles, negativeGradient):
            # Added a small step size to prevent overshooting and help convergence
            step_size = 0.005 # You might need to tune this value
            tempJointAngles += [jointAngle + (gradientParameter + random.uniform(0, 0.01)) * (step_size + random.uniform(0, 0.01))]
        jointAngles = tempJointAngles

        zeroVector = endPointDifferenceMap(jointAngles)
        # print(f"Current error: {zeroVector:.4f}") # Uncomment for debugging convergence
        iteration += 1

    if iteration == max_iterations:
        print("WARNING: moveToPosition did not converge within max iterations.")

    return jointAngles

def convertModelAnglesToServoAngles(modelJointAngles):
    '''Convert model joint angles to servo joint angles.
    
    Input:
        modelJointAngles: List of model joint angles in radians.

    Output:
        servoJointAngles: List of servo joint angles in degrees
    '''

    servoJointAngles = [None] * len(modelJointAngles)
     
    for i, modelJointAngle in enumerate(modelJointAngles):
        
          
        match i:
            case 0 | 2 | 3 | 4:
                servoJointAngles[i] = math.degrees(modelJointAngle) + 90
            case 1:
                servoJointAngles[i] = math.degrees(modelJointAngle) * -1
            case 5: # Assuming servo 6 (index 5) is for the gripper and might have a different mapping
                servoJointAngles[i] = math.degrees(modelJointAngle) # Or adjust as per gripper's specific mapping
            case _:
                # Handle cases for more than 6 joints if applicable, or raise an error
                pass

    return servoJointAngles

def convertServoAnglesToModelAngles(servoJointAngles):
    '''Convert servo joint angles to model joint angles.
    
    Input:
        servoJointAngles: List of servo joint angles in degrees.

    Output:
        modelAngles: List of model joint angles in radians.
    '''
     
    modelAngles = [None] * len(servoJointAngles)

    for i, servoAngle in enumerate(servoJointAngles):
         
        match i:
            case 0 | 2 | 3 | 4:
                modelAngles[i] = math.radians(servoAngle - 90)
            case 1:
                modelAngles[i] = math.radians(servoAngle * -1)
            case 5: # Assuming servo 6 (index 5) is for the gripper
                modelAngles[i] = math.radians(servoAngle) # Or adjust as per gripper's specific mapping
            case _:
                # Handle cases for more than 6 joints if applicable, or raise an error
                pass
    
    return modelAngles

def motionPlan(linkVectors, jointAxis, jointAngles, endPos):
    '''Generate motion plan for the arm.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        endPos: Matrix object representing the end position.

    Output:
        motionPlanAngles: List of servo joint angles representing the motion plan.

    Notes:
        This function uses moveToPosition(), which is slow in comparison to the new goToPos(). It is 
        also made irrelevant by the traceShape() function.
    '''
    startingPos = getPosition(linkVectors, jointAxis, jointAngles)[-1]

    x = np.linspace(startingPos.mat[0][0], endPos.mat[0][0], 10)
    y = np.linspace(startingPos.mat[1][0], endPos.mat[1][0], 10)
    z = np.linspace(startingPos.mat[2][0], endPos.mat[2][0], 10)

    motionPlanAngles = []
    for i, xPos in enumerate(x):
        jointAngles = moveToPosition(linkVectors, jointAxis, jointAngles, matrix([[xPos], [y[i]], [z[i]]]))
        motionPlanAngles.append(convertModelAnglesToServoAngles(jointAngles))
    
    return motionPlanAngles

def laserProjectionMap(linkVectors, jointAxis, jointAngles, xDistanceToPlane=93.25):
    '''Generate laser projection map. Find world coordinates of laser being projected onto
    a plane in the ZY plane.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        xDistanceToPlane: Distance to the plane in the x-direction, default 93.25.

    Output:
        laserEndPoint: Matrix object representing the end point of the laser.
    
    Notes:
        This function is used as a map for numdifftools later.
    '''

    endLinks = getPosition(linkVectors, jointAxis, jointAngles)

    # Get the laser unit vector
    laserVector = endLinks[-1] - endLinks[-2]
    laserUnitVector = matrix(laserVector.mat/np.linalg.norm(laserVector.mat))

    # Find the distance between the ZY plane and the end effector in the x component
    distanceFromPlane = xDistanceToPlane - endLinks[-1].mat[0][0]

    # Find the scaling coefficient that makes the x component of the laserUnitVector equal to the distanceFromPlane
    # Handle division by zero if laserUnitVector.mat[0][0] is zero or very close to zero
    if abs(laserUnitVector.mat[0][0]) < 1e-9: # A small epsilon to avoid division by zero
        # If the laser is pointing perpendicular to the plane, it won't intersect at xDistanceToPlane
        # You might want to raise an error or return a specific value
        print("WARNING: Laser is parallel to the target plane in X-direction. Cannot project.")
        return None # Or handle this case as appropriate for your application
    
    scalingConstant = distanceFromPlane/laserUnitVector.mat[0][0]

    # Create the final laserVector value, which is the unit vector scaled by the scaling constant found above
    laserVector = scalingConstant * laserUnitVector

    # Find the laserEndPoint, which is the laserVector + the end position of the arm
    laserEndPoint = endLinks[-1] + laserVector

    return laserEndPoint

def pointAtPositionInZYPlane(linkVectors, jointAxis, jointAngles, xDistanceToPlane, yPos, zPos, distanceFromPos=0.2, jointLimits=[(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90), (-90, 90)]): # Updated to 6 limits
    '''Point a laser at a position in the ZY plane.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        xDistanceToPlane: Distance to the plane in the x-direction.
        yPos: Y-coordinate of the position.
        zPos: Z-coordinate of the position.
        distanceFromPos: Acceptable amount of displacement from requested position, default 0.2.
        jointLimits: List of tuples representing joint limits, default [(-90, 90), (-180, 0), (-90, 90), (-90, 90), (-90, 90), (-90, 90)].

    Output:
        jointAngles: List of joint angles representing the arm's position.

    Notes:
        System is very unstable, and it may be useful to use a much larger distanceFromPos value.
    '''
    desiredEndVector = matrix([[xDistanceToPlane],[yPos],[zPos]])

    # Function defining the map to position of the laser on the whiteboard (ZY plane)
    def laserProjectionMapDifference(jointAngles_local): # Renamed jointAngles to jointAngles_local
        
        # Force joint limits to be respected in the mapping
        for i, jointLimit in enumerate(jointLimits):
            # Ensure we don't try to access a joint limit that doesn't exist for the current joint index
            if i < len(jointAngles_local):
                if min(jointLimit) > math.degrees(jointAngles_local[i]):
                    jointAngles_local[i] = math.radians(min(jointLimit))
                elif max(jointLimit) < math.degrees(jointAngles_local[i]):
                    jointAngles_local[i] = math.radians(max(jointLimit))

        laserEndPoint = laserProjectionMap(linkVectors, jointAxis, jointAngles_local, xDistanceToPlane)
        if laserEndPoint is None: # Handle case where laser projection fails
            return float('inf') # Return a large error to indicate failure

        # Create a final output whos value is the magnitude of the difference between the laser's end point and its desired endpoint
        diff = laserEndPoint - desiredEndVector
        absDifference = diff.norm()

        return absDifference
    
    laserProjectionMapGradient = nd.Gradient(laserProjectionMapDifference)

    zeroVector = laserProjectionMapDifference(jointAngles)

    # Added a safeguard to prevent infinite loops if convergence is not met
    max_iterations = 1000
    iteration = 0

    while zeroVector > distanceFromPos and iteration < max_iterations:
        gradient = laserProjectionMapGradient(jointAngles)
        negativeGradient = gradient * -1

        tempJointAngles = []
        for jointAngle, gradientParameter in zip(jointAngles, negativeGradient):
            # Random numbers are to escape saddle points, and proportionally symmetric geometry
            # Increased step size slightly for potentially faster convergence, but be careful with stability
            step_size = 0.001 # Original was 0.0005, increased slightly
            tempJointAngles += [jointAngle + (gradientParameter + random.uniform(0, 0.00001)) * (step_size + random.uniform(0, 0.00001))] 
        jointAngles = tempJointAngles

        zeroVector = laserProjectionMapDifference(jointAngles)
        # print(f"Current laser error: {zeroVector:.4f}") # Uncomment for debugging convergence
        iteration += 1

    if iteration == max_iterations:
        print("WARNING: pointAtPositionInZYPlane did not converge within max iterations.")

    return jointAngles

def laserMotionPlan(linkVectors, jointAxis, jointAngles, desiredY, desiredZ, xDist=93.25):
    '''Generate motion plan for laser.
    
    Input:
        linkVectors: List of matrix objects representing link vectors.
        jointAxis: List of characters representing joint rotation axes.
        jointAngles: List of joint angles in radians.
        desiredY: Desired Y-coordinate for the laser.
        desiredZ: Desired Z-coordinate for the laser.
        xDist: Distance to the plane in the x-direction, default 93.25.

    Output:
        motionPlanAngles: List of servo joint angles representing the motion plan.
    '''

    startingLaserEndPoint = laserProjectionMap(linkVectors, jointAxis, jointAngles, xDist)
    if startingLaserEndPoint is None: # Handle case where initial projection fails
        print("Error: Initial laser projection failed. Cannot generate motion plan.")
        return []

    y = np.linspace(startingLaserEndPoint.mat[1][0], desiredY, 10)
    z = np.linspace(startingLaserEndPoint.mat[2][0], desiredZ, 10)

    motionPlanAngles = []
    for i, yPos in enumerate(y):
        # Pass the current jointAngles to pointAtPositionInZYPlane for iterative refinement
        jointAngles = pointAtPositionInZYPlane(linkVectors, jointAxis, jointAngles, xDist, yPos, z[i])
        motionPlanAngles.append(convertModelAnglesToServoAngles(jointAngles))

    return motionPlanAngles


# --- End of functions from kinematics.py ---


class InverseKinematics:
    """
    Class for Inverse Kinematics calculations for the LeArm robot,
    utilizing the provided kinematic functions.
    """

    def __init__(self):
        """
        Initializes the InverseKinematics solver by loading arm attributes
        from the YAML file using buildArm().
        """
        self.linkVectors, self.jointAxis, self.initial_jointAngles_rad = buildArm()
        print("InverseKinematics initialized with arm attributes from arm_attributes.yaml.")

        # Define joint limits in degrees based on your LeArm's physical constraints.
        # These are crucial for ensuring valid and safe movements.
        # The values here are examples and should be verified for your specific LeArm.
        # The order should correspond to the order of joints in your model.
        # Assuming 6 servos for the LeArm, typically:
        # Servo 1 (Base/Yaw), Servo 2 (Shoulder/Pitch), Servo 3 (Elbow/Pitch),
        # Servo 4 (Wrist/Pitch), Servo 5 (Wrist/Roll), Servo 6 (Gripper/Open-Close)
        # IMPORTANT: Ensure the number of entries here matches the number of joints
        # defined in your arm_attributes.yaml file (e.g., 6).
        self.joint_limits_deg = {
            1: (0, 180),   # Base (yaw)
            2: (0, 180),   # Shoulder (pitch)
            3: (0, 180),   # Elbow (pitch)
            4: (0, 180),   # Wrist (pitch)
            5: (0, 180),   # Wrist (roll)
            6: (0, 180)    # Gripper (open/close) - often mapped differently
        }
        print("Joint limits defined (please verify for your LeArm model).")


    def get_current_end_effector_position(self, current_servo_angles_deg):
        """
        Calculates the current end-effector position based on the current servo angles.
        This uses your `getPosition` function.

        Args:
            current_servo_angles_deg (list): List of current servo angles in degrees.

        Returns:
            matrix: A matrix object representing the end-effector's 3D position.
        """
        # Convert servo angles (degrees) to model angles (radians)
        model_angles_rad = convertServoAnglesToModelAngles(current_servo_angles_deg)
        
        # Get the end-effector position using your forward kinematics function
        # Assuming the last link's end point is the end-effector
        end_effector_pos_matrix = getPosition(self.linkVectors, self.jointAxis, model_angles_rad)[-1]
        return end_effector_pos_matrix


    def solve_ik(self, target_x, target_y, target_z, current_servo_angles_deg=None):
        """
        Solves the inverse kinematics problem to find the joint angles
        required to reach the specified target (x, y, z) position.

        This method leverages your `goToPos` function for finding the joint angles.

        Args:
            target_x (float): Desired X-coordinate for the end-effector.
            target_y (float): Desired Y-coordinate for the end-effector.
            target_z (float): Desired Z-coordinate for the end-effector.
            current_servo_angles_deg (list, optional): An initial guess for the servo angles in degrees.
                                                        If None, `self.initial_jointAngles_rad` (from YAML)
                                                        will be used as a starting point.

        Returns:
            list: A list of calculated servo joint angles in degrees,
                  or None if the target position is unreachable or a solution
                  cannot be found.
        """
        desired_end_pos_matrix = matrix([[target_x], [target_y], [target_z]])

        # Convert current servo angles (if provided) to model angles (radians)
        if current_servo_angles_deg is not None:
            starting_model_angles_rad = convertServoAnglesToModelAngles(current_servo_angles_deg)
        else:
            starting_model_angles_rad = self.initial_jointAngles_rad

        print(f"Solving IK for target ({target_x}, {target_y}, {target_z}) from starting model angles: {np.degrees(starting_model_angles_rad).round(2)}")

        try:
            # Use your goToPos function to find the model angles
            solved_model_angles_rad = goToPos(self.linkVectors, self.jointAxis, starting_model_angles_rad, desired_end_pos_matrix)
            
            # Convert the solved model angles (radians) back to servo angles (degrees)
            solved_servo_angles_deg = convertModelAnglesToServoAngles(solved_model_angles_rad)

            # Optional: Add a check here to ensure solved_servo_angles_deg are within the LeArm's physical limits
            # For example:
            # for i, angle in enumerate(solved_servo_angles_deg):
            #     servo_id = i + 1 # Assuming 1-indexed servo IDs
            #     min_limit, max_limit = self.joint_limits_deg.get(servo_id, (0, 180)) # Default if not found
            #     if not (min_limit <= angle <= max_limit):
            #         print(f"WARNING: Solved angle for servo {servo_id} ({angle:.2f} deg) is outside its physical limit ({min_limit}-{max_limit} deg).")
            #         # You might choose to clamp the angle or return None here.

            print(f"IK solution found (servo angles in degrees): {np.array(solved_servo_angles_deg).round(2)}")
            return solved_servo_angles_deg

        except ValueError as e:
            print(f"IK solution failed: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during IK solving: {e}")
            return None

    def plan_laser_motion(self, desiredY, desiredZ, xDist=93.25, current_servo_angles_deg=None):
        """
        Generates a motion plan for the laser to trace a path on the ZY plane.
        Leverages your `laserMotionPlan` function.

        Args:
            desiredY (float): Desired final Y-coordinate for the laser.
            desiredZ (float): Desired final Z-coordinate for the laser.
            xDist (float, optional): Distance to the plane in the x-direction. Defaults to 93.25.
            current_servo_angles_deg (list, optional): Initial servo angles in degrees.

        Returns:
            list: A list of lists, where each inner list contains servo joint angles
                  (in degrees) for each step of the motion plan. Returns empty list if plan fails.
        """
        if current_servo_angles_deg is not None:
            starting_model_angles_rad = convertServoAnglesToModelAngles(current_servo_angles_deg)
        else:
            starting_model_angles_rad = self.initial_jointAngles_rad

        print(f"Planning laser motion to Y={desiredY}, Z={desiredZ} at X={xDist}")
        motion_plan = laserMotionPlan(self.linkVectors, self.jointAxis, starting_model_angles_rad, desiredY, desiredZ, xDist)
        
        if not motion_plan:
            print("Laser motion plan could not be generated.")
        else:
            print(f"Laser motion plan generated with {len(motion_plan)} steps.")
        return motion_plan


if __name__ == "__main__":
    # --- Example Usage of the Integrated IK Class ---

    # Initialize the IK solver. It will load arm attributes from arm_attributes.yaml.
    ik_solver = InverseKinematics()

    # Initialize the LeArm controller
    try:
        arm = LeArm(debug=True)
        print("LeArm controller initialized.")
        # Set initial positions for the servos to a known state (e.g., mid-range)
        # This is important because the arm doesn't report its current position.
        # These values should correspond to a safe, known starting configuration.
        initial_arm_servo_angles = [90, 90, 90, 90, 90, 90] # Example: mid-range for all servos
        print(f"Moving arm to initial safe position: {initial_arm_servo_angles} degrees.")
        arm.servoMove([arm.servo1, arm.servo2, arm.servo3, arm.servo4, arm.servo5, arm.servo6], initial_arm_servo_angles, time=1500)
        time.sleep(1.5) # Give the arm time to reach the position

    except Exception as e:
        print(f"ERROR: Could not initialize LeArm controller. Physical arm movement will be skipped. Error: {e}")
        print("Please ensure the LeArm is connected and the necessary drivers/permissions are set up.")
        # If arm initialization fails, we can still run IK calculations, just not move the physical arm.
        arm = None 
        # Use initial_arm_servo_angles for IK calculations even if physical arm is not connected
        initial_arm_servo_angles = [90, 90, 90, 90, 90, 90] # Fallback for IK starting point


    # --- Define your desired target position in 3D space ---
    # These coordinates are in the same units as your linkVectors in arm_attributes.yaml
    # (e.g., cm or mm). Make sure they are reachable by your arm.
    target_x = 15.0 # Example: 15 cm forward from the base
    target_y = 5.0   # Example: 5 cm to the side
    target_z = 20.0  # Example: 20 cm above the base

    print(f"\n--- Testing IK to reach position: (X={target_x}, Y={target_y}, Z={target_z}) ---")
    solved_angles_deg = ik_solver.solve_ik(target_x, target_y, target_z, current_servo_angles_deg=initial_arm_servo_angles)

    if solved_angles_deg:
        print(f"IK solution: {np.array(solved_angles_deg).round(2)} degrees.")
        if arm:
            servos_to_control = [arm.servo1, arm.servo2, arm.servo3, arm.servo4, arm.servo5, arm.servo6]
            if len(solved_angles_deg) == len(servos_to_control):
                print(f"Moving LeArm to calculated angles...")
                arm.servoMove(servos_to_control, solved_angles_deg, time=2000)
                print("LeArm move command sent. Waiting for movement to complete...")
                time.sleep(2.5) # Wait for the arm to complete its movement
                print("Movement complete.")
            else:
                print("Error: Mismatch between the number of solved angles and the number of physical servos.")
        else:
            print("LeArm controller not available for physical movement.")
    else:
        print("Failed to find IK solution for the target position.")

    # --- Test Laser Motion Planning ---
    print(f"\n--- Testing Laser Motion Plan ---")
    desired_laser_y = 10.0 # Example target Y on the plane
    desired_laser_z = 25.0 # Example target Z on the plane
    laser_plane_x_dist = 93.25 # From your kinematics.py

    laser_motion_plan = ik_solver.plan_laser_motion(desired_laser_y, desired_laser_z, laser_plane_x_dist, current_servo_angles_deg=initial_arm_servo_angles)

    if laser_motion_plan:
        print(f"Generated laser motion plan with {len(laser_motion_plan)} steps.")
        if arm:
            servos_to_control = [arm.servo1, arm.servo2, arm.servo3, arm.servo4, arm.servo5, arm.servo6]
            for i, step_angles in enumerate(laser_motion_plan):
                if len(step_angles) == len(servos_to_control):
                    print(f"Executing laser motion step {i+1}: {np.array(step_angles).round(2)} degrees.")
                    arm.servoMove(servos_to_control, step_angles, time=500) # Move faster for steps
                    # time.sleep(0.5) # Wait for arm to reach position if needed
                else:
                    print(f"Error in laser motion plan step {i+1}: Mismatch in number of angles and servos.")
            print("Laser motion plan execution complete.")
        else:
            print("LeArm controller not available for physical movement.")
    else:
        print("Failed to generate laser motion plan.")

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

