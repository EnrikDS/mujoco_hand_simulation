# scripts/run_viewer.py
import os, sys, argparse
import numpy as np
import mujoco, mujoco.viewer
import matplotlib.pyplot as plt
plt.ion()    
import time                  
import matplotlib.pyplot as plt           # NEW
from mujoco_grasp.utils import (
    site_pose,
    ForceGridPlot  )               
from mujoco_grasp.sensors import bind_grid
from mujoco_grasp.control import ForceController, FingertipPositionController, TrajectoryGenerator

print("MuJoCo:", mujoco.__version__, "Python:", sys.version)

# ---------------- CLI flags ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--plot-forces", action="store_true",
                    help="Show live Fx/Fy/Fz time plots (slower).")
parser.add_argument("--plot-window", type=float, default=10.0,
                    help="Time window [s] shown in the force plots.")
parser.add_argument("--active-ui", action="store_true",
                    help="Use interactive viewer (sliders/keyboard).")
args = parser.parse_args()

PLOT_WINDOW_S = args.plot_window

# ---------------- Paths ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path   = os.path.join(script_dir, "../mj/full_combined.xml")
print(xml_path)

# ---------------- Load model/data ----------------
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# Print model info for debugging
print(f"Model has {model.nq} positions, {model.nu} actuators")

# Static name→joint table for UR10e and Allegro
name_to_joint = {
    # UR10e
    "shoulder_pan": "shoulder_pan_joint",
    "shoulder_lift":"shoulder_lift_joint",
    "elbow":        "elbow_joint",
    "wrist_1":      "wrist_1_joint",
    "wrist_2":      "wrist_2_joint",
    "wrist_3":      "wrist_3_joint",
    # Allegro
    "ffa0": "ffj0","ffa1": "ffj1","ffa2": "ffj2","ffa3": "ffj3",
    "mfa0": "mfj0","mfa1": "mfj1","mfa2": "mfj2","mfa3": "mfj3",
    "rfa0": "rfj0","rfa1": "rfj1","rfa2": "rfj2","rfa3": "rfj3",
    "tha0": "thj0","tha1": "thj1","tha2": "thj2","tha3": "thj3",
}

# Global variables for control and state
act_joint_qadr = None
ctrl_hold = None
desired_qpos = None
desired_qvel = None
reset_flag = False

GRID_SPECS = {
    "palm_grid":   (3, 40, 20),
    "ff_tip_grid": (3, 20, 20),
    "mf_tip_grid": (3, 20, 20),
    "rf_tip_grid": (3, 20, 20),
    "th_tip_grid": (3, 20, 20),
}


def initialize_control_mapping():
    """Initialize the actuator-to-joint mapping for control"""
    global act_joint_qadr, ctrl_hold
    
    # ---------------- Hold-position control map (UR + hand) ----------------
    n_actuators = model.nu
    act_joint_qadr = np.full(n_actuators, -1, dtype=int)
    trntype = np.asarray(model.actuator_trntype)
    trnid   = np.asarray(model.actuator_trnid)

    # Sanity checks (helpful if XML changes later)
    assert trntype.ndim == 1 and trntype.size == n_actuators, f"unexpected actuator_trntype shape: {trntype.shape}"
    assert trnid.ndim   == 2 and trnid.shape[0] == n_actuators and trnid.shape[1] >= 1, f"unexpected actuator_trnid shape: {trnid.shape}"

    # Build mapping actuator -> joint qpos address
    for aid in range(n_actuators):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        jname = name_to_joint.get(aname)
        if jname:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                act_joint_qadr[aid] = int(model.jnt_qposadr[jid])
                print(f"Mapped actuator '{aname}' -> joint '{jname}' -> qpos[{act_joint_qadr[aid]}]")
            else:
                print(f"[WARN] Joint '{jname}' not found for actuator '{aname}'")
        else:
            print(f"[WARN] No joint mapping for actuator '{aname}'")
    
    # Initialize control targets array
    ctrl_hold = np.zeros(n_actuators, dtype=float)

def get_home_positions():
    """Get the home position dictionaries"""
    # UR home positions
    UR_HOME = {
        "shoulder_pan_joint":  0.0,
        "shoulder_lift_joint": -0.88,
        "elbow_joint":          0.44,
        "wrist_1_joint":        2.64,
        "wrist_2_joint":        0.0628,
        "wrist_3_joint":        -0.628,
    }
    
    # Allegro home positions (more conservative values to avoid weird configurations)
    HAND_HOME = {
        "ffj0": 0.00, "mfj0": 0.00, "rfj0": 0.00, "thj0": 0.50,  # Reduced thumb rotation
        "ffj1": 0.30, "ffj2": 0.40, "ffj3": 0.30,  # More conservative finger positions
        "mfj1": 0.30, "mfj2": 0.40, "mfj3": 0.30,
        "rfj1": 0.30, "rfj2": 0.40, "rfj3": 0.30,
        "thj1": 0.20, "thj2": 0.30, "thj3": 0.20,  # More conservative thumb
    }
    
    return UR_HOME, HAND_HOME

def apply_joint_positions(UR_HOME, HAND_HOME):
    """Apply the joint positions to the model"""
    # Build a full-length qpos vector
    q_home_full = np.zeros(model.nq, dtype=float)
    
    # Apply UR joint positions
    for jn, val in UR_HOME.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            qpos_idx = model.jnt_qposadr[jid]
            q_home_full[qpos_idx] = float(val)
            print(f"Set {jn} (joint {jid}) to {val} at qpos[{qpos_idx}]")
        else:
            print(f"[ERROR] Joint '{jn}' not found!")
    
    # Apply hand joint positions
    for jn, val in HAND_HOME.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            qpos_idx = model.jnt_qposadr[jid]
            # Check joint limits
            jnt_range = model.jnt_range[jid]
            if jnt_range[0] <= val <= jnt_range[1]:
                q_home_full[qpos_idx] = float(val)
                print(f"Set {jn} (joint {jid}) to {val} at qpos[{qpos_idx}], range: {jnt_range}")
            else:
                print(f"[WARN] Value {val} for {jn} outside range {jnt_range}, skipping")
        else:
            print(f"[WARN] Joint '{jn}' not found!")
    
    # Set the joint positions
    data.qpos[:] = q_home_full
    
    return q_home_full

def initialize_box_position():
    """Initialize box in a safe position away from robot"""
    BOX_BODY = "grocery_box"
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "grocery_box_free")
    if box_joint_id >= 0:
        box_qpos_addr = model.jnt_qposadr[box_joint_id]
        # Place box in a safe location initially (above and away from robot)
        data.qpos[box_qpos_addr:box_qpos_addr+3] = [0.5, 0.5, 1.5]  # x, y, z
        data.qpos[box_qpos_addr+3:box_qpos_addr+7] = [1.0, 0.0, 0.0, 0.0]  # quaternion (w, x, y, z)
        print(f"Initialized box at safe position: {data.qpos[box_qpos_addr:box_qpos_addr+3]}")

def apply_hold_control():
    """Apply hold control to keep robot in position"""
    for aid in range(model.nu):
        qadr = act_joint_qadr[aid]
        if qadr >= 0:
            data.ctrl[aid] = ctrl_hold[aid]

def settle_robot(UR_HOME, steps=2000):
    """Settle the robot into position with control"""
    print("Settling robot into position...")
    for step in range(steps):
        # Apply control
        apply_hold_control()
        mujoco.mj_step(model, data)
        
        # Print progress and check convergence
        if step % 400 == 399:
            print(f"  Step {step+1}/{steps}")
            # Check how close we are to target
            max_error = 0.0
            for jn in UR_HOME.keys():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    current_pos = data.qpos[model.jnt_qposadr[jid]]
                    target_pos = UR_HOME[jn]
                    error = abs(current_pos - target_pos)
                    max_error = max(max_error, error)
            print(f"    Max joint error: {max_error:.4f}")
            
            # If we've converged enough, break early
            if max_error < 0.01:  # 0.01 radians ≈ 0.57 degrees
                print(f"    Converged early at step {step+1}")
                break

def place_box_correctly():
    """Place the box at the correct position relative to the robot"""
    print("Placing box...")
    WRIST_SITE = "attachment_site"

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WRIST_SITE)
    if sid >= 0:
        w_pos = data.site_xpos[sid].copy()
        w_R = data.site_xmat[sid].reshape(3,3)
        
        # Place box slightly above and in front of the palm
        n_out = -w_R[:, 2]  # -Z axis of wrist frame
        half_thickness = 0.02
        gap = 0.0003
        box_center = w_pos + (half_thickness + gap) * n_out + np.array([0.0, -0.1, 0.02])  # Slightly above wrist
        
        print(f"Final wrist position: {w_pos}")
        print(f"Placing box at: {box_center}")
        
        # Set box position
        box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "grocery_box_free")
        if box_joint_id >= 0:
            box_qpos_addr = model.jnt_qposadr[box_joint_id]
            data.qpos[box_qpos_addr:box_qpos_addr+3] = box_center
            data.qpos[box_qpos_addr+3:box_qpos_addr+7] = [1.0, 0.0, 0.0, 0.0]
            
            # Zero out velocities for the box
            box_dof_addr = model.jnt_dofadr[box_joint_id]
            data.qvel[box_dof_addr:box_dof_addr+6] = 0.0
            
            mujoco.mj_forward(model, data)
            print(f"Box successfully placed at: {data.qpos[box_qpos_addr:box_qpos_addr+3]}")
            
            # Run a few more steps to ensure the box settles properly
            print("Letting box settle...")
            for _ in range(100):
                apply_hold_control()
                mujoco.mj_step(model, data)
            
            print(f"Box final position: {data.qpos[box_qpos_addr:box_qpos_addr+3]}")
    else:
        print(f"[ERROR] Site '{WRIST_SITE}' not found!")

def setup_robot_configuration():
    """Complete robot and box setup routine"""
    global ctrl_hold, desired_qpos, desired_qvel
    
    print("Setting up robot configuration...")
    
    # Reset data first
    mujoco.mj_resetData(model, data)
    
    # Get home positions
    UR_HOME, HAND_HOME = get_home_positions()
    
    # Apply joint positions
    apply_joint_positions(UR_HOME, HAND_HOME)
    
    # Initialize box in safe position
    initialize_box_position()
    
    # Forward kinematics to update all positions
    mujoco.mj_forward(model, data)
    
    # Set control targets
    n_actuators = model.nu
    for aid in range(n_actuators):
        qadr = act_joint_qadr[aid]
        if qadr >= 0:
            ctrl_hold[aid] = float(data.qpos[qadr])
            data.ctrl[aid] = ctrl_hold[aid]  # Set control immediately

    print(f"Control targets set: {ctrl_hold}")
    
    # Settle the robot
    settle_robot(UR_HOME)
    
    print("Robot settled. Final joint positions:")
    for jn in UR_HOME.keys():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            current_pos = data.qpos[model.jnt_qposadr[jid]]
            target_pos = UR_HOME[jn]
            print(f"  {jn}: target={target_pos:.3f}, current={current_pos:.3f}, error={abs(current_pos-target_pos):.4f}")
    
    # Place the box correctly
    place_box_correctly()
    
    # Store the desired configuration
    desired_qpos = data.qpos.copy()
    desired_qvel = data.qvel.copy()
    
    print("Robot configuration setup complete!")
    return UR_HOME, HAND_HOME

def reset_to_desired_state():
    """Reset robot and box to our desired configuration"""
    global desired_qpos, desired_qvel, reset_flag
    
    print("Resetting to desired configuration...")
    
    # If we don't have a stored configuration, set it up
    if desired_qpos is None or desired_qvel is None:
        print("No stored configuration found, setting up initial configuration...")
        setup_robot_configuration()
        reset_flag = False
        return
    
    # Restore the desired state
    data.qpos[:] = desired_qpos.copy()
    data.qvel[:] = desired_qvel.copy()
    
    # Set controls immediately after reset
    apply_hold_control()
    
    # Forward kinematics to update all positions
    mujoco.mj_forward(model, data)
    
    reset_flag = False
    print("Reset complete")

def check_if_reset_needed():
    """Check if we need to reset based on robot configuration"""
    global reset_flag
    
    if desired_qpos is None:
        return False
    
    # Check if current state differs significantly from desired state
    # Focus on key joints that should maintain their position
    UR_HOME, _ = get_home_positions()
    
    max_deviation = 0.0
    for jn in UR_HOME.keys():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            qpos_idx = model.jnt_qposadr[jid]
            current_pos = data.qpos[qpos_idx]
            desired_pos = desired_qpos[qpos_idx]
            deviation = abs(current_pos - desired_pos)
            max_deviation = max(max_deviation, deviation)
    
    # If deviation is large and time is near zero, likely a reset occurred
    if max_deviation > 0.5 and data.time < 0.1:  # 0.5 rad ~ 28 degrees
        reset_flag = True
        return True
    
    return False

# Initialize everything
initialize_control_mapping()
UR_HOME, HAND_HOME = setup_robot_configuration()

# Initialize force controller for hand torque control
force_controller = ForceController(model, max_joint_speed=20.0)

# Initialize controller
controller = FingertipPositionController(model, max_joint_torque=0.5)
# Initialize trajectory generator  
traj_gen = TrajectoryGenerator()
traj_gen.detect_palm_center(data)
 # Get initial positions
initial_positions = controller.get_fingertip_positions(data)

# Generate grasp trajectory
trajectory = traj_gen.generate_grasp_trajectory(initial_positions, grasp_duration=3.0)

# Control loop (this would go in your main simulation loop)
start_time = data.time + 5


    

# ---------------- Optional force plots ----------------
dt = model.opt.timestep

plotter = None
sensor_defs = {}   # name -> dict(adr, dim, shape)

if args.plot_forces:
    # bind once
    ff_tip_sid, ff_tip_adr, ff_tip_dim, ff_tip_shape = bind_grid(model, "ff_tip_grid", GRID_SPECS["ff_tip_grid"])
    mf_tip_sid, mf_tip_adr, mf_tip_dim, mf_tip_shape = bind_grid(model, "mf_tip_grid", GRID_SPECS["mf_tip_grid"])
    rf_tip_sid, rf_tip_adr, rf_tip_dim, rf_tip_shape = bind_grid(model, "rf_tip_grid", GRID_SPECS["rf_tip_grid"])
    th_tip_sid, th_tip_adr, th_tip_dim, th_tip_shape = bind_grid(model, "th_tip_grid", GRID_SPECS["th_tip_grid"])
    palm_sid,   palm_adr,   palm_dim,   palm_shape   = bind_grid(model, "palm_grid",   GRID_SPECS["palm_grid"])

    sensor_defs = {
        "palm": {"adr": palm_adr, "dim": palm_dim, "shape": palm_shape},
        "ff":   {"adr": ff_tip_adr, "dim": ff_tip_dim, "shape": ff_tip_shape},
        "mf":   {"adr": mf_tip_adr, "dim": mf_tip_dim, "shape": mf_tip_shape},
        "rf":   {"adr": rf_tip_adr, "dim": rf_tip_dim, "shape": rf_tip_shape},
        "th":   {"adr": th_tip_adr, "dim": th_tip_dim, "shape": th_tip_shape},
    }

    plotter = ForceGridPlot(labels=list(sensor_defs.keys()),
                        horizon_s=PLOT_WINDOW_S, dt=dt)
# ---------------- Viewer loop with robust reset handling ----------------
STEP_VIS = 10
k = 0

if args.active_ui:
    # ---- ACTIVE UI: viewer steps the sim; sliders work ----
    with mujoco.viewer.launch(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        model.vis.scale.framelength = 0.08
        model.vis.scale.framewidth  = 0.003

        while viewer.is_running():
            
            time.sleep(0.2)
            
            # keep feeding your hold targets so the arm doesn't drift
            apply_hold_control()
            
            # Apply force control demo if active
         

            # no mj_step() here — the viewer is stepping internally

            if args.plot_forces and plotter is not None:
                k += 1
                if k % STEP_VIS == 0:
                    tnow = data.time
                    values = {}
                    for label, s in sensor_defs.items():
                        flat = data.sensordata[s["adr"]: s["adr"] + s["dim"]]
                        arr  = flat.reshape(s["shape"])
                        Fz = arr[0].sum()
                        Fx = arr[1].sum() if arr.shape[0] > 1 else 0.0
                        Fy = arr[2].sum() if arr.shape[0] > 2 else 0.0
                        values[label] = (Fx, Fy, Fz)
                    plotter.append(tnow, values)
                    plotter.refresh(t_window=PLOT_WINDOW_S)
                    plt.pause(0.001)

            viewer.sync()

else:
    # ---- PASSIVE: your loop steps the sim; no UI sliders ----
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        model.vis.scale.framelength = 0.08
        model.vis.scale.framewidth  = 0.003
        
        while viewer.is_running():
            if (data.time - start_time < 5.0) & (data.time  - start_time > 0.0):  # Run for 3 seconds
                # Get desired positions at current time
                t = data.time - start_time
                desired_pos = trajectory(t)
                print(f"t={t:.2f}s, desired positions: {desired_pos}")
                # Apply position control
                controller.control_to_positions(data, desired_pos)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Optional: print progress
            if int(t * 10) % 10 == 0:  # Every 0.1 seconds
                current_pos = controller.get_fingertip_positions(data)
                print(f"t={t:.1f}s - Forefinger at: {current_pos.get('forefinger', [0,0,0])}")

                if args.plot_forces and plotter is not None:
                    k += 1
                    if k % STEP_VIS == 0:
                        tnow = data.time
                        values = {}
                        for label, s in sensor_defs.items():
                            flat = data.sensordata[s["adr"]: s["adr"] + s["dim"]]
                            arr  = flat.reshape(s["shape"])
                            Fz = arr[0].sum()
                            Fx = arr[1].sum() if arr.shape[0] > 1 else 0.0
                            Fy = arr[2].sum() if arr.shape[0] > 2 else 0.0
                            values[label] = (Fx, Fy, Fz)
                        plotter.append(tnow, values)
                        plotter.refresh(t_window=PLOT_WINDOW_S)
                        plt.pause(0.001)

            viewer.sync()