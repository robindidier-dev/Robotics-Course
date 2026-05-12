"""
Implementation of the framework of Siciliano & Slotine. 
It defines the Task and TaskLevel classes, as well as the main Framework class.

Written by ourselves.
"""

import numpy as np

class Task:
    """ Represents a control task with its associated function, jacobian, and desired velocity. """
    def __init__(self, name: str, function, jacobian, xdot_des: np.ndarray):
        self.name = name
        self.function = function      
        self.jacobian = jacobian      
        self.xdot_des = xdot_des      

class TaskLevel:
    """ Represents a task at a specific priority level, storing 
        the task itself, its projector, and the computed joint velocities. """
    def __init__(self, task: Task):
        self.task = task
        self.P = None       
        self.qdot = None    

class Framework:
    def __init__(self, robot):
        self.robot = robot
        self.tasks = []
        self.use_dls = False  
        self.damping = 1    

    def add_task(self, task: Task):
        self.tasks.append(TaskLevel(task))

    def pseudo_inverse_dls(self, J: np.ndarray) -> np.ndarray:
        """ Damped least-squares computation. """
        m, n = J.shape
        I = np.eye(m)
        return J.T @ np.linalg.inv(J @ J.T + (self.damping**2) * I)

    def solve(self) -> np.ndarray:
        qdot_prev = np.zeros(self.robot.n)
        P_prev = np.eye(self.robot.n)

        for level in self.tasks:
            J = level.task.jacobian(self.robot.q)
            xdot_des = level.task.xdot_des

            # Jacobian projection in the null space of previous tasks
            J_proj = J @ P_prev

            if self.use_dls:
                Jsharp_vel = self.pseudo_inverse_dls(J_proj)
            else:
                Jsharp_vel = np.linalg.pinv(J_proj)

            # Jsharp_vel -> for velocity control so DLS
            # Jsharp_proj -> for projector update so as precise as possible 
            # But the same in the framwork of Siciliano & Slotine
            Jsharp_proj = np.linalg.pinv(J_proj, rcond=1e-5)

            # qdot computation 
            qdot = qdot_prev + Jsharp_vel @ (xdot_des - J @ qdot_prev)
            
            # Projector update
            P = P_prev @ (np.eye(self.robot.n) - Jsharp_proj @ J_proj)

            level.qdot = qdot
            level.P = P

            qdot_prev = qdot
            P_prev = P

        return qdot_prev