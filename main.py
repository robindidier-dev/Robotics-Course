"""
Main file for the robot simulation. 

It initializes the robot and framework, and loops to update the 
robot's state and the visualization.

File written by AI.
"""

import sys
import numpy as np
from collections import deque
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from robot import Robot
from framework import Framework
import tasks_instance as ti

robot = Robot(arms=[1, 1, 1, 1, 1, 1, 1], q=[-0.1]*7)
framework = Framework(robot)
target = np.array([3.0, 0.0])
obs_pos = np.array([2.5, 0])
obs_radius_securite = 1.0
hist_len = 200

use_posture, use_obstacle, use_tilt = False, False, False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Robot")
        self.setGeometry(100, 100, 1300, 750)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_xlim(-2, 7); self.ax.set_ylim(-3, 6); self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        self.line_outer, = self.ax.plot([], [], '-', color='#333333', lw=8, solid_capstyle='round')
        
        self.joints_outer = [Circle((0,0), 0.11, color='#333333', zorder=3) for _ in range(robot.n + 1)]
        self.joints_inner = [Circle((0,0), 0.07, color='#AAAAAA', zorder=4) for _ in range(robot.n + 1)]
        for c in self.joints_outer + self.joints_inner: self.ax.add_patch(c)

        self.obs_patch = Circle(obs_pos, 0.7, color='#ff6666', alpha=0.0, zorder=1)
        self.safe_patch = Circle(obs_pos, obs_radius_securite, edgecolor='#ff6666', facecolor='none', linestyle='--', alpha=0.0, zorder=1)
        self.ax.add_patch(self.obs_patch); self.ax.add_patch(self.safe_patch)
        
        self.target_dot, = self.ax.plot([], [], 'x', color='#2ca02c', markersize=12, markeredgewidth=3)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        layout.addWidget(self.canvas, stretch=2)

        right_panel = QVBoxLayout()
        self.hist_tasks = deque([0]*hist_len, maxlen=hist_len)
        self.hist_error = deque([0]*hist_len, maxlen=hist_len)
        self.hist_qdot = [deque([0]*hist_len, maxlen=hist_len) for _ in range(robot.n)]

        self.plot_tasks = pg.PlotWidget(title="Number of tasks")
        self.curve_tasks = self.plot_tasks.plot(pen=pg.mkPen(color='orange', width=3))
        
        self.plot_error = pg.PlotWidget(title="Tracking error")
        self.curve_error = self.plot_error.plot(pen=pg.mkPen(color='orange', width=3))
        
        self.plot_qdot = pg.PlotWidget(title="Velocities")
        self.curves_qdot = []
        n = robot.n
        for i in range(n):
            t = i / (n - 1)
            if t < 0.5:
                val = t * 2
                color = (int(128 * val), int(100 + 28 * val), int(255 - 127 * val))
            else:
                val = (t - 0.5) * 2
                color = (int(128 + 127 * val), int(128 + 12 * val), int(128 - 128 * val))
            self.curves_qdot.append(self.plot_qdot.plot(pen=pg.mkPen(color=color, width=2)))
        
        for p in [self.plot_tasks, self.plot_error, self.plot_qdot]:
            p.showGrid(x=True, y=True)
            right_panel.addWidget(p)
        
        btn_layout = QHBoxLayout()
        self.btn_tilt = QPushButton("Tilt : OFF"); self.btn_tilt.clicked.connect(self.toggle_tilt)
        self.btn_dls = QPushButton("DLS : OFF"); self.btn_dls.clicked.connect(self.toggle_dls)
        self.btn_posture = QPushButton("Posture : OFF"); self.btn_posture.clicked.connect(self.toggle_posture)
        self.btn_obs = QPushButton("Obstacle : OFF"); self.btn_obs.clicked.connect(self.toggle_obs)
        for b in [self.btn_tilt, self.btn_dls, self.btn_posture, self.btn_obs]: btn_layout.addWidget(b)
        right_panel.addLayout(btn_layout)
        layout.addLayout(right_panel, stretch=1)

        self.timer = QTimer(); self.timer.timeout.connect(self.update_loop); self.timer.start(20)

    def toggle_tilt(self):
        global use_tilt; use_tilt = not use_tilt
        self.btn_tilt.setText(f"Tilt : {'ON' if use_tilt else 'OFF'}")

    def toggle_dls(self):
        framework.use_dls = not framework.use_dls
        self.btn_dls.setText(f"DLS : {'ON' if framework.use_dls else 'OFF'}")

    def toggle_posture(self):
        global use_posture; use_posture = not use_posture
        self.btn_posture.setText(f"Posture : {'ON' if use_posture else 'OFF'}")

    def toggle_obs(self):
        global use_obstacle; use_obstacle = not use_obstacle
        self.obs_patch.set_alpha(0.6 if use_obstacle else 0.0)
        self.safe_patch.set_alpha(0.8 if use_obstacle else 0.0)
        self.btn_obs.setText(f"Obstacle : {'ON' if use_obstacle else 'OFF'}")

    def on_click(self, event):
        global target
        if event.xdata is not None: target = np.array([event.xdata, event.ydata])

    def update_loop(self):
        framework.tasks = []
        if use_obstacle:
            threatened_joints = [(k, np.linalg.norm(robot.get_joint_pos(k) - obs_pos)) 
                                 for k in range(1, robot.n + 1) if np.linalg.norm(robot.get_joint_pos(k) - obs_pos) < obs_radius_securite]
            if threatened_joints: framework.add_task(ti.create_multiple_obstacle_task(robot, threatened_joints, obs_pos, obs_radius_securite))
        
        if use_tilt: framework.add_task(ti.create_tilt_task(robot, target_angle=-np.pi/2))
        framework.add_task(ti.create_tracking_task(robot, target))
        if use_posture: framework.add_task(ti.create_posture_task(robot))

        qdot = framework.solve()
        robot.q += qdot * 0.01

        self.hist_tasks.append(len(framework.tasks))
        self.hist_error.append(np.linalg.norm(target - robot.forward_kinematics()))
        for i in range(robot.n): self.hist_qdot[i].append(qdot[i])

        self.curve_tasks.setData(list(self.hist_tasks))
        self.curve_error.setData(list(self.hist_error))
        for i, curve in enumerate(self.curves_qdot): curve.setData(list(self.hist_qdot[i]))

        pts = [(0,0)] + [robot.get_joint_pos(k+1) for k in range(robot.n)]
        self.line_outer.set_data([p[0] for p in pts], [p[1] for p in pts])
        
        for i, p in enumerate(pts):
            self.joints_outer[i].set_center(p)
            self.joints_inner[i].set_center(p)
            
        self.target_dot.set_data([target[0]], [target[1]])
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())