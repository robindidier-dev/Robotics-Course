"""
This file defines specific task instances for the robot control framework, 
such as obstacle avoidance, tracking, tilt control, and posture maintenance. 

Written by ourselves.
"""

import numpy as np
from framework import Task


def create_multiple_obstacle_task(robot, threatened_joints, obs_pos, obs_radius_securite):
    """
    Creates a hierarchical task for obstacle avoidance using a geometric approach.
    Each threatened joint is treated as a separate constraint row.
    """
    
    def f(q): 
        # Evaluation of current distances for each threatened joint
        dists = []
        for k, _ in threatened_joints:
            pos = robot.get_joint_pos(k, q)
            dists.append(np.linalg.norm(pos - obs_pos))
        return np.array(dists)
        
    def J(q):
        # Initialize Jacobian matrix
        J_matrix = np.zeros((len(threatened_joints), robot.n))
        
        for idx, (k, _) in enumerate(threatened_joints):
            pos = robot.get_joint_pos(k, q)
            dist = np.linalg.norm(pos - obs_pos)
            
            # Singular protection: Avoid division by zero if joint is on obstacle center
            if dist > 1e-5:
                # Get the full geometric Jacobian for joint 'k'
                J_k = robot.get_joint_jacobian(k, q) 
                
                # Normal escape vector (unit vector pointing from obstacle to joint)
                n_vec = (pos - obs_pos) / dist 
                
                # Projection: Only keep the component of joint motion that increases distance.
                # This corresponds to the gradient of the distance function.
                J_matrix[idx, :] = n_vec @ J_k 
                
        return J_matrix

    # Task Error: Positive if joint is inside the safety radius (intrusion detected)
    errors = [obs_radius_securite - dist for _, dist in threatened_joints]
    
    # Desired velocity (xdot): We want to move away at a speed proportional to the intrusion.
    # A gain of 10.0 ensures a reactive push-back behavior.
    xdot_des = np.array([10.0 * e for e in errors]) 
    
    return Task("Obstacles", f, J, xdot_des)

    
def create_tracking_task(robot, target_pos):
    def f(q): return robot.forward_kinematics(q)
    def J(q): return robot.jacobian(q)
    error = target_pos - robot.forward_kinematics(robot.q)
    xdot_des = 15.0 * error  
    norm = np.linalg.norm(xdot_des)
    if norm > 15.0: xdot_des = (xdot_des / norm) * 15.0
    return Task("Tracking", f, J, xdot_des)

def create_tilt_task(robot, target_angle):
    def f(q): return np.array([np.sum(q)])
    def J(q): return np.ones((1, robot.n))
    error = target_angle - np.sum(robot.q)
    return Task("Tilt", f, J, np.array([5.0 * error]))

def create_posture_task(robot):
    def f(q): return np.array([0.5 * np.sum(q**2)])
    def J(q): return np.array([q])
    error = 0.0 - (0.5 * np.sum(robot.q**2))
    return Task("Posture", f, J, np.array([5 * error]))