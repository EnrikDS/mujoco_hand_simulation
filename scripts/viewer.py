# scripts/run_viewer.py
import time, numpy as np, mujoco, mujoco.viewer
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "../mj/combined.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# Small joint dither so you see motion: move UR10e shoulder a bit if actuator exists
def actuator_id(name):
    try: return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    except: return -1

# Example: if actuators are named like "shoulder_pan" etc — print names and pick yours
print("Actuators:", [model.names[model.name_actuatoradr[i]] for i in range(model.nu)])

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running():
        t = time.time() - t0
        # If your arm uses position actuators, set data.ctrl[...] here.
        # data.ctrl[:] = 0  # or small sine by index after you identify actuators
        mujoco.mj_step(model, data)
        viewer.sync()
