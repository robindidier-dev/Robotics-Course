"""
Class representing a planar robot arm with n revolute joints. 

It provides methods for computing :
- Forward kinematics (joints position)
- Jacobian (mapping joint velocities to end-effector velocity)

Written by ourselves.
"""

import numpy as np

class Robot:
    def __init__(self, arms, q):
        self.arms = np.array(arms)
        self.q = np.array(q)
        self.n = len(arms)

    def forward_kinematics(self, q=None):
        """ Returns end-effector position. """
        return self.get_joint_pos(self.n, q)

    def jacobian(self, q=None):
        """ End-effector jacobian. """
        return self.get_joint_jacobian(self.n, q)

    def get_joint_pos(self, k, q=None):
        """ (x, y) position of the kth articulation. """
        if q is None: q = self.q
        angle, x, y = 0.0, 0.0, 0.0
        for i in range(k):
            angle += q[i]
            x += self.arms[i] * np.cos(angle)
            y += self.arms[i] * np.sin(angle)
        return np.array([x, y])

    def get_joint_jacobian(self, k, q=None):
        """ Jacobian of the kth articulation. """
        if q is None: q = self.q
        J = np.zeros((2, self.n))
        angle_sum = np.cumsum(q)
        for joint in range(k):
            dx, dy = 0.0, 0.0
            for i in range(joint, k):
                angle = angle_sum[i]
                dx -= self.arms[i] * np.sin(angle)
                dy += self.arms[i] * np.cos(angle)
            J[0, joint] = dx
            J[1, joint] = dy
        return J