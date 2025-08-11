# scripts/run_viewer.py
import time, numpy as np, mujoco, mujoco.viewer
import os
import numpy as np
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "../mj/full_combined.xml")
#xml_path = os.path.join(script_dir, "../assets/ur10e/ur10e.xml")
print(xml_path)
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

q_home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0] + [0]*16)
data.qpos[:len(q_home)] = q_home

# Map actuator -> joint qpos index, then set ctrl = qpos
ur_joint_names = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                  "wrist_1_joint","wrist_2_joint","wrist_3_joint"]

for i, jname in enumerate(ur_joint_names):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
        qadr = model.jnt_qposadr[jid]
        if i < model.nu:                # actuator index aligned with your UR ordering
            data.ctrl[i] = data.qpos[qadr]
     
mujoco.mj_forward(model, data)

# Example: if actuators are named like "shoulder_pan" etc — print names and pick yours
print("Actuators:", [model.names[model.name_actuatoradr[i]] for i in range(model.nu)])
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
if site_id < 0:
    raise RuntimeError("Site 'attachment_site' not found. Check the UR XML.")

pos = data.site_xpos[site_id].copy()
mat9 = data.site_xmat[site_id].copy()  # 3x3 in row-major (flattened)
R = mat9.reshape(3,3)

quat = np.empty(4)
mujoco.mju_mat2Quat(quat, R.flatten(order='C'))

vals = " ".join(f"{x:.6f}" for x in data.qpos)  # this is exactly 29 numbers now
print(f'<key name="home_full" qpos="{vals}"/>')

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    model.vis.scale.framelength = 0.08
    model.vis.scale.framewidth  = 0.003
    while viewer.is_running():
        t = time.time() - t0
        # If your arm uses position actuators, set data.ctrl[...] here.
        # data.ctrl[:] = 0  # or small sine by index after you identify actuators
        mujoco.mj_step(model, data)
        viewer.sync()
