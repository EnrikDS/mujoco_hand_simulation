import mujoco
import numpy as np
from mujoco_grasp.utils import (site_pose, TimePlot)

class FingertipPositionController:
    """
    Position controller for fingertip trajectory tracking using Jacobian-based control.
    Each fingertip follows a desired 3D trajectory with joint torque limits.
    """
    
    def __init__(self, model, max_joint_speed=2.0, max_joint_torque=1.0):
        """
        Initialize fingertip position controller.
        
        Args:
            model: MuJoCo model
            max_joint_speed: Maximum allowed joint speed (rad/s)
            max_joint_torque: Maximum torque per joint (N⋅m)
        """
        self.model = model
        self.max_joint_speed = max_joint_speed
        self.max_joint_torque = max_joint_torque
        
        # Hand joint organization
        self.finger_joints = {
            'forefinger': ['ffj0', 'ffj1', 'ffj2', 'ffj3'],
            'middle': ['mfj0', 'mfj1', 'mfj2', 'mfj3'], 
            'ring': ['rfj0', 'rfj1', 'rfj2', 'rfj3'],
            'thumb': ['thj0', 'thj1', 'thj2', 'thj3']
        }
        
        # Fingertip sites
        self.finger_sites = {
            'forefinger': 'ff_tip_grid_site',
            'middle': 'mf_tip_grid_site',
            'ring': 'rf_tip_grid_site',
            'thumb': 'th_tip_grid_site'
        }
        
        # Control gains (tune these)
        self.kp = 100.0  # Position gain
        self.kd = 10.0   # Velocity gain
        
        # Create actuator mappings
        self.joint_to_actuator = {}
        name_to_joint = {
            "ffa0": "ffj0", "ffa1": "ffj1", "ffa2": "ffj2", "ffa3": "ffj3",
            "mfa0": "mfj0", "mfa1": "mfj1", "mfa2": "mfj2", "mfa3": "mfj3", 
            "rfa0": "rfj0", "rfa1": "rfj1", "rfa2": "rfj2", "rfa3": "rfj3",
            "tha0": "thj0", "tha1": "thj1", "tha2": "thj2", "tha3": "thj3"
        }
        
        for aid in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            jname = name_to_joint.get(aname)
            if jname:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    self.joint_to_actuator[jname] = aid
        
        # Pre-compute joint indices for Jacobian
        self._joint_indices = {}
        for finger_name, joint_names in self.finger_joints.items():
            indices = []
            for joint_name in joint_names:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    dof_adr = model.jnt_dofadr[joint_id]
                    indices.append(dof_adr)
            self._joint_indices[finger_name] = indices
        
        print(f"Position controller initialized for {len(self.finger_sites)} fingertips")
        print(f"Max joint torque: {self.max_joint_torque} N⋅m")
        
    def compute_fingertip_jacobians(self, data):
        """
        Compute position Jacobians (3x4) for each fingertip.
        Only translational part needed for position control.
        
        Returns:
            dict: Position Jacobians for each finger
        """
        jacobians = {}
        
        for finger_name, site_name in self.finger_sites.items():
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                continue
                
            body_id = self.model.site_bodyid[site_id]
            
            # Full Jacobians (world frame)
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, data, jac_pos, jac_rot, data.site_xpos[site_id], body_id)
            
            # Extract finger-specific columns (only position Jacobian)
            if finger_name in self._joint_indices:
                joint_indices = self._joint_indices[finger_name]
                if len(joint_indices) == 4:
                    jac_finger = jac_pos[:, joint_indices]  # 3x4 matrix
                    
                    jacobians[finger_name] = {
                        'jacobian': jac_finger,
                        'site_pos': data.site_xpos[site_id].copy(),
                        'joint_indices': joint_indices
                    }
        
        return jacobians
    
    def get_fingertip_positions(self, data):
        """Get current fingertip positions."""
        positions = {}
        
        for finger_name, site_name in self.finger_sites.items():
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                positions[finger_name] = data.site_xpos[site_id].copy()
                
        return positions
    
    def get_fingertip_velocities(self, data):
        """Compute fingertip velocities using Jacobian."""
        jacobians = self.compute_fingertip_jacobians(data)
        velocities = {}
        
        for finger_name, jac_data in jacobians.items():
            joint_indices = jac_data['joint_indices']
            J = jac_data['jacobian']
            
            # Get joint velocities for this finger
            q_dot = np.array([data.qvel[idx] for idx in joint_indices])
            
            # Compute Cartesian velocity: v = J * q_dot
            v_tip = J @ q_dot
            velocities[finger_name] = v_tip
            
        return velocities
    
    def control_to_positions(self, data, desired_positions, desired_velocities=None):
        """
        Control fingertips to desired positions using PD control in Cartesian space.
        
        Args:
            data: MuJoCo data
            desired_positions: dict with target positions for each finger
                {
                    'forefinger': [x, y, z],
                    'middle': [x, y, z],
                    ...
                }
            desired_velocities: dict with target velocities (optional)
                {
                    'forefinger': [vx, vy, vz],
                    ...
                }
        """
        # Get current states
        current_positions = self.get_fingertip_positions(data)
        current_velocities = self.get_fingertip_velocities(data)
        jacobians = self.compute_fingertip_jacobians(data)
        
        if desired_velocities is None:
            desired_velocities = {name: np.zeros(3) for name in desired_positions.keys()}
        
        for finger_name in desired_positions.keys():
            if finger_name not in jacobians or finger_name not in current_positions:
                continue
                
            # Position and velocity errors
            pos_error = np.array(desired_positions[finger_name]) - current_positions[finger_name]
            vel_error = np.array(desired_velocities[finger_name]) - current_velocities[finger_name]
            
            # PD control law in Cartesian space
            F_desired = self.kp * pos_error + self.kd * vel_error
            
            # Convert to joint torques using Jacobian transpose
            J = jacobians[finger_name]['jacobian']
            tau = J.T @ F_desired  # 4x1 vector
            
            # Apply torque limits
            tau = np.clip(tau, -self.max_joint_torque, self.max_joint_torque)
            
            # Apply to actuators with speed limits
            joint_names = self.finger_joints[finger_name]
            for i, joint_name in enumerate(joint_names):
                if joint_name in self.joint_to_actuator:
                    aid = self.joint_to_actuator[joint_name]
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    
                    if jid >= 0:
                        # Check speed limit
                        qvel_addr = self.model.jnt_dofadr[jid]
                        joint_vel = abs(data.qvel[qvel_addr])
                        
                        if joint_vel > self.max_joint_speed:
                            data.ctrl[aid] = 0.0  # Stop if over speed limit
                        else:
                            data.ctrl[aid] = tau[i]
    
    def set_control_gains(self, kp, kd):
        """Update PD control gains."""
        self.kp = kp
        self.kd = kd
        print(f"Updated gains: kp={kp}, kd={kd}")


class TrajectoryGenerator:
    """
    Generates trajectories for fingertips to grasp objects in the palm center.
    """
    
    def __init__(self, palm_center=None):
        """
        Initialize trajectory generator.
        
        Args:
            palm_center: [x, y, z] position of palm center (auto-detected if None)
        """
        self.palm_center = palm_center or [0.0, 0.0, 0.0]  # Will be updated
        
    def detect_palm_center(self, data):
        """
        Auto-detect palm center from current fingertip positions.
        
        Args:
            data: MuJoCo data
        """
        controller = FingertipPositionController.__new__(FingertipPositionController)
        controller.finger_sites = {
            'forefinger': 'ff_tip_grid_site',
            'middle': 'mf_tip_grid_site', 
            'ring': 'rf_tip_grid_site',
            'thumb': 'th_tip_grid_site'
        }
        
        positions = []
        for finger_name, site_name in controller.finger_sites.items():
            site_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                positions.append(data.site_xpos[site_id])
        
        if positions:
            # Palm center is roughly the centroid of fingertips, moved toward wrist
            centroid = np.mean(positions, axis=0)
            # Move toward wrist (assume wrist is behind palm in -x direction)
            self.palm_center = centroid + np.array([-0.05, 0.0, -0.03])
        
        print(f"Detected palm center: {self.palm_center}")
        return self.palm_center
    
    def generate_grasp_trajectory(self, current_positions, grasp_duration=3.0, approach_height=0.08):
        """
        Generate smooth trajectories from current fingertip positions to grasp points.
        
        Args:
            current_positions: dict with current fingertip positions
            grasp_duration: Total time for grasp motion (seconds)
            approach_height: Height above palm center for approach points (m)
            
        Returns:
            function: trajectory(t) -> desired_positions dict
        """
        # Define target positions around palm center
        target_positions = {
            'forefinger': self.palm_center + np.array([0.02, 0.02, 0.01]),   # Slightly forward and out
            'middle': self.palm_center + np.array([0.025, 0.0, 0.01]),       # Most forward
            'ring': self.palm_center + np.array([0.02, -0.02, 0.01]),        # Forward and in  
            'thumb': self.palm_center + np.array([-0.025, 0.03, 0.01])       # Back and out (opposition)
        }
        
        # Approach points (above targets)
        approach_positions = {}
        for finger_name, target_pos in target_positions.items():
            approach_positions[finger_name] = target_pos + np.array([0.0, 0.0, approach_height])
        
        def trajectory(t):
            """
            Generate desired positions at time t.
            
            Phase 1 (0 to 50%): Move to approach points
            Phase 2 (50% to 100%): Move from approach to grasp points
            """
            t = np.clip(t / grasp_duration, 0.0, 1.0)  # Normalize time
            
            desired_positions = {}
            
            for finger_name in current_positions.keys():
                if finger_name not in target_positions:
                    continue
                
                start_pos = np.array(current_positions[finger_name])
                approach_pos = approach_positions[finger_name]
                target_pos = target_positions[finger_name]
                
                if t <= 0.5:
                    # Phase 1: Move to approach point
                    phase_t = t * 2.0  # 0 to 1
                    # Smooth interpolation using cubic
                    s = 3 * phase_t**2 - 2 * phase_t**3
                    current_target = start_pos + s * (approach_pos - start_pos)
                else:
                    # Phase 2: Move from approach to grasp
                    phase_t = (t - 0.5) * 2.0  # 0 to 1
                    # Smooth interpolation
                    s = 3 * phase_t**2 - 2 * phase_t**3
                    current_target = approach_pos + s * (target_pos - approach_pos)
                
                desired_positions[finger_name] = current_target
            
            return desired_positions
        
        return trajectory
    
    def generate_simple_trajectory(self, current_positions, grasp_duration=2.0):
        """
        Simple direct trajectory to palm center (no intermediate points).
        
        Args:
            current_positions: dict with current fingertip positions
            grasp_duration: Total time for motion (seconds)
            
        Returns:
            function: trajectory(t) -> desired_positions dict
        """
        # Target positions closer to palm center
        target_positions = {
            'forefinger': self.palm_center + np.array([0.015, 0.015, 0.005]),
            'middle': self.palm_center + np.array([0.02, 0.0, 0.005]),
            'ring': self.palm_center + np.array([0.015, -0.015, 0.005]),
            'thumb': self.palm_center + np.array([-0.02, 0.02, 0.005])
        }
        
        def trajectory(t):
            t = np.clip(t / grasp_duration, 0.0, 1.0)
            
            # Smooth S-curve interpolation
            s = 3 * t**2 - 2 * t**3
            
            desired_positions = {}
            for finger_name in current_positions.keys():
                if finger_name in target_positions:
                    start_pos = np.array(current_positions[finger_name])
                    target_pos = target_positions[finger_name]
                    desired_positions[finger_name] = start_pos + s * (target_pos - start_pos)
            
            return desired_positions
        
        return trajectory
# Usage example
def example_grasp_control(model, data):
    """
    Example of how to use the position controller for grasping.
    """
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
    start_time = data.time
    
    while data.time - start_time < 5.0:  # Run for 3 seconds
        # Get desired positions at current time
        t = data.time - start_time
        desired_pos = trajectory(t)
        
        # Apply position control
        controller.control_to_positions(data, desired_pos)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Optional: print progress
        if int(t * 10) % 10 == 0:  # Every 0.1 seconds
            current_pos = controller.get_fingertip_positions(data)
            print(f"t={t:.1f}s - Forefinger at: {current_pos.get('forefinger', [0,0,0])}")

class ForceController:
    """
    Force controller class for finger joints with torque control, speed limiting, and Jacobian computation.
    Controls 16 finger joints (4 fingers × 4 joints each) with velocity-based safety limits.
    """
    
    def __init__(self, model, max_joint_speed=1.0):
        """
        Initialize the force controller.
        
        Args:
            model: MuJoCo model
            max_joint_speed: Maximum allowed joint speed in rad/s
        """
        self.model = model
        self.max_joint_speed = max_joint_speed
        
        # Hand joint organization: 4 fingers × 4 joints each
        self.finger_joints = {
            'forefinger': ['ffj0', 'ffj1', 'ffj2', 'ffj3'],
            'middle': ['mfj0', 'mfj1', 'mfj2', 'mfj3'], 
            'ring': ['rfj0', 'rfj1', 'rfj2', 'rfj3'],
            'thumb': ['thj0', 'thj1', 'thj2', 'thj3']
        }
        
        # Fingertip sites for Jacobian computation
        self.finger_sites = {
            'forefinger': 'ff_tip_grid_site',
            'middle': 'mf_tip_grid_site',
            'ring': 'rf_tip_grid_site',
            'thumb': 'th_tip_grid_site'
        }
        
        # Create mapping from joint names to actuator indices
        self.joint_to_actuator = {}
        self.actuator_to_joint = {}
        
        # Map actuators to joints
        name_to_joint = {
            "ffa0": "ffj0", "ffa1": "ffj1", "ffa2": "ffj2", "ffa3": "ffj3",
            "mfa0": "mfj0", "mfa1": "mfj1", "mfa2": "mfj2", "mfa3": "mfj3", 
            "rfa0": "rfj0", "rfa1": "rfj1", "rfa2": "rfj2", "rfa3": "rfj3",
            "tha0": "thj0", "tha1": "thj1", "tha2": "thj2", "tha3": "thj3"
        }
        
        for aid in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            jname = name_to_joint.get(aname)
            if jname:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    self.joint_to_actuator[jname] = aid
                    self.actuator_to_joint[aid] = jname
        
        # Pre-compute joint indices for efficiency
        self._joint_indices = {}
        for finger_name, joint_names in self.finger_joints.items():
            indices = []
            for joint_name in joint_names:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    dof_adr = model.jnt_dofadr[joint_id]
                    indices.append(dof_adr)
            self._joint_indices[finger_name] = indices
        
        print(f"ForceController initialized with {len(self.joint_to_actuator)} finger joints")
        print(f"Max joint speed: {self.max_joint_speed} rad/s")
        print(f"Finger sites: {list(self.finger_sites.values())}")
    
    def compute_finger_jacobians(self, data):
        """
        Compute Jacobian matrices for each fingertip in tip reference frame.
        Returns only the finger-specific Jacobians (6x4 for each finger).
        
        Args:
            data: MuJoCo data
            
        Returns:
            dict: Jacobians for each finger with keys:
                - 'jacobian': 6x4 full Jacobian (position + rotation)
                - 'position_jac': 3x4 position Jacobian
                - 'rotation_jac': 3x4 rotation Jacobian
                - 'site_pos': fingertip position
                - 'site_mat': fingertip rotation matrix
        """
        jacobians = {}
        
        for finger_name, site_name in self.finger_sites.items():
            # Get site ID
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                print(f"Warning: Site '{site_name}' not found for {finger_name}")
                continue
                
            # Get site body ID
            body_id = self.model.site_bodyid[site_id]
            
            # Allocate Jacobian matrices
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            
            # Compute Jacobians in world frame
            mujoco.mj_jac(self.model, data, jac_pos, jac_rot, data.site_xpos[site_id], body_id)
            
            # Get joint indices for this finger
            if finger_name not in self._joint_indices or len(self._joint_indices[finger_name]) != 4:
                print(f"Warning: Invalid joint indices for {finger_name}")
                continue
                
            joint_indices = self._joint_indices[finger_name]
            
            # Extract finger-specific columns
            jac_pos_finger = jac_pos[:, joint_indices]  # 3x4
            jac_rot_finger = jac_rot[:, joint_indices]  # 3x4
            jac_full_finger = np.vstack([jac_pos_finger, jac_rot_finger])  # 6x4
            
            # Transform to fingertip reference frame
            R_world_to_tip = data.site_xmat[site_id].reshape(3, 3).T
            
            # Transform Jacobian to tip frame
            R_6x6 = np.block([
                [R_world_to_tip, np.zeros((3, 3))],
                [np.zeros((3, 3)), R_world_to_tip]
            ])
            
            jac_local = R_6x6 @ jac_full_finger
            
            jacobians[finger_name] = {
                'jacobian': jac_local,              # 6x4
                'position_jac': jac_local[:3, :],   # 3x4  
                'rotation_jac': jac_local[3:, :],   # 3x4
                'site_pos': data.site_xpos[site_id].copy(),
                'site_mat': data.site_xmat[site_id].reshape(3, 3).copy(),
                'joint_names': self.finger_joints[finger_name]
            }
        
        return jacobians
    
    def compute_torques_from_forces(self, data, desired_forces):
        """
        Compute joint torques required to achieve desired fingertip forces.
        
        Args:
            data: MuJoCo data
            desired_forces: dict with desired forces/torques for each finger:
                {
                    'forefinger': [fx, fy, fz, mx, my, mz],  # 6D wrench in tip frame
                    'middle': [fx, fy, fz, mx, my, mz],
                    ...
                }
        
        Returns:
            dict: Joint torques for each finger:
                {
                    'forefinger': [t0, t1, t2, t3],
                    'middle': [t0, t1, t2, t3],
                    ...
                }
        """
        jacobians = self.compute_finger_jacobians(data)
        joint_torques = {}
        
        for finger_name, F_desired in desired_forces.items():
            if finger_name not in jacobians:
                continue
                
            F_desired = np.array(F_desired)
            if F_desired.shape[0] != 6:
                print(f"Warning: Expected 6D wrench for {finger_name}, got {F_desired.shape[0]}")
                continue
            
            # Use pseudo-inverse to compute torques: τ = J^+ * F
            J = jacobians[finger_name]['jacobian']
            J_pinv = np.linalg.pinv(J)
            tau = J_pinv @ F_desired
            
            joint_torques[finger_name] = tau.tolist()
        
        return joint_torques
    
    def compute_forces_from_torques(self, data, finger_torques):
        """
        Compute fingertip forces from joint torques using Jacobian transpose.
        
        Args:
            data: MuJoCo data
            finger_torques: dict with joint torques for each finger:
                {
                    'forefinger': [t0, t1, t2, t3],
                    'middle': [t0, t1, t2, t3],
                    ...
                }
        
        Returns:
            dict: Fingertip wrenches for each finger:
                {
                    'forefinger': {'force': [fx, fy, fz], 'torque': [mx, my, mz]},
                    'middle': {'force': [fx, fy, fz], 'torque': [mx, my, mz]},
                    ...
                }
        """
        jacobians = self.compute_finger_jacobians(data)
        fingertip_wrenches = {}
        
        for finger_name, torques in finger_torques.items():
            if finger_name not in jacobians:
                continue
                
            tau = np.array(torques)
            if tau.shape[0] != 4:
                print(f"Warning: Expected 4 torques for {finger_name}, got {tau.shape[0]}")
                continue
            
            # Compute wrench: F = J^T * tau
            J = jacobians[finger_name]['jacobian']
            wrench = J.T @ tau
            
            fingertip_wrenches[finger_name] = {
                'force': wrench[:3].tolist(),   # 3D force in tip frame
                'torque': wrench[3:].tolist()   # 3D torque in tip frame
            }
        
        return fingertip_wrenches
    
    def apply_force_control(self, data, desired_forces):
        """
        High-level force control: compute and apply torques for desired fingertip forces.
        
        Args:
            data: MuJoCo data
            desired_forces: dict with desired forces for each finger (see compute_torques_from_forces)
        """
        # Compute required torques
        torques = self.compute_torques_from_forces(data, desired_forces)
        
        # Apply torque control with speed limiting
        self.apply_torque_control(data, torques)
    
    def apply_torque_control(self, data, finger_torques):
        """
        Apply torque control to finger joints with speed limiting.
        
        Args:
            data: MuJoCo data
            finger_torques: Dictionary with finger torques organized as:
                {
                    'forefinger': [t0, t1, t2, t3],
                    'middle': [t0, t1, t2, t3], 
                    'ring': [t0, t1, t2, t3],
                    'thumb': [t0, t1, t2, t3]
                }
        """
        for finger_name, joint_names in self.finger_joints.items():
            if finger_name in finger_torques:
                torques = finger_torques[finger_name]
                
                if len(torques) != 4:
                    print(f"[WARN] Expected 4 torques for {finger_name}, got {len(torques)}")
                    continue
                
                for i, joint_name in enumerate(joint_names):
                    if joint_name in self.joint_to_actuator:
                        aid = self.joint_to_actuator[joint_name]
                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        
                        if jid >= 0:
                            # Get joint velocity
                            qvel_addr = self.model.jnt_dofadr[jid]
                            joint_vel = abs(data.qvel[qvel_addr])
                            
                            # Apply speed limiting: zero torque if over max speed
                            if joint_vel > self.max_joint_speed:
                                data.ctrl[aid] = 0.0
                                if joint_vel > self.max_joint_speed * 1.1:
                                    print(f"Speed limit hit on {joint_name}: {joint_vel:.3f} > {self.max_joint_speed}")
                            else:
                                data.ctrl[aid] = torques[i]
    
    def get_jacobian_info(self, data):
        """
        Get summary information about current Jacobian matrices.
        Useful for debugging and monitoring singularities.
        
        Returns:
            dict: Information about each finger's Jacobian
        """
        jacobians = self.compute_finger_jacobians(data)
        info = {}
        
        for finger_name, jac_data in jacobians.items():
            J = jac_data['jacobian']
            cond_num = np.linalg.cond(J)
            
            info[finger_name] = {
                'condition_number': cond_num,
                'is_singular': cond_num > 100,  # Threshold for singularity warning
                'tip_position': jac_data['site_pos'].tolist(),
                'jacobian_shape': J.shape
            }
        
        return info

# Example usage functions (can be removed if not needed)
def example_force_control(controller, data):
    """Example of how to use force control"""
    
    # Example 1: Apply downward forces on index and thumb
    desired_forces = {
        'forefinger': [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],  # 2N downward
        'thumb': [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]        # 1N downward
    }
    
    controller.apply_force_control(data, desired_forces)
    print("Applied force control")

def example_torque_control(controller, data):
    """Example of direct torque control"""
    
    # Example 2: Apply small torques to close fingers
    torques = {
        'forefinger': [0.1, 0.2, 0.1, 0.05],
        'middle': [0.1, 0.2, 0.1, 0.05],
        'ring': [0.1, 0.2, 0.1, 0.05],
        'thumb': [0.1, 0.1, 0.1, 0.05]
    }
    
    controller.apply_torque_control(data, torques)
    
    # Check what forces this produces
    forces = controller.compute_forces_from_torques(data, torques)
    for finger, wrench in forces.items():
        print(f"{finger}: F={wrench['force']}, T={wrench['torque']}")

def monitor_jacobians(controller, data):
    """Example of monitoring Jacobian health"""
    
    info = controller.get_jacobian_info(data)
    
    for finger_name, finger_info in info.items():
        cond = finger_info['condition_number']
        singular = finger_info['is_singular']
        
        status = "SINGULAR" if singular else "OK"
        print(f"{finger_name}: Condition={cond:.2f} [{status}]")
    
    def get_joint_states(self, data):
        """
        Get current joint positions and velocities organized by finger.
        
        Returns:
            Dictionary with structure:
            {
                'forefinger': {'positions': [q0, q1, q2, q3], 'velocities': [qd0, qd1, qd2, qd3]},
                'middle': {'positions': [q0, q1, q2, q3], 'velocities': [qd0, qd1, qd2, qd3]},
                'ring': {'positions': [q0, q1, q2, q3], 'velocities': [qd0, qd1, qd2, qd3]},
                'thumb': {'positions': [q0, q1, q2, q3], 'velocities': [qd0, qd1, qd2, qd3]}
            }
        """
        states = {}
        
        for finger_name, joint_names in self.finger_joints.items():
            positions = []
            velocities = []
            
            for joint_name in joint_names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid >= 0:
                    qpos_addr = self.model.jnt_qposadr[jid]
                    qvel_addr = self.model.jnt_dofadr[jid]
                    
                    positions.append(data.qpos[qpos_addr])
                    velocities.append(data.qvel[qvel_addr])
                else:
                    positions.append(0.0)
                    velocities.append(0.0)
            
            states[finger_name] = {
                'positions': positions,
                'velocities': velocities
            }
        
        return states


def place_free_body_at_pose(model, data, body_name, pos_w, R_w):
    """Place a free-jointed body at world pose (pos_w, R_w) and zero its velocities."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"Body '{body_name}' not found")
    jadr = model.body_jntadr[bid]
    if model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Body '{body_name}' does not have a free joint")
    qadr = model.jnt_qposadr[jadr]
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R_w.flatten(order='C'))       # accepts 3x3 on 3.3.5
    data.qpos[qadr:qadr+3]     = pos_w
    data.qpos[qadr+3:qadr+7]   = quat
    data.qvel[qadr:qadr+6]     = 0.0
    mujoco.mj_forward(model, data)


def apply_hold_control(model, data, act_joint_qadr, ctrl_hold):
    """Apply hold control to keep robot in position"""
    for aid in range(model.nu):
        qadr = act_joint_qadr[aid]
        if qadr >= 0:
            data.ctrl[aid] = ctrl_hold[aid]

def initialize_control_mapping(model, name_to_joint):
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

def apply_joint_positions(UR_HOME, HAND_HOME, model, data):
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

def setup_robot_configuration(model, data):
    """Complete robot and box setup routine"""
    global ctrl_hold, desired_qpos, desired_qvel
    
    print("Setting up robot configuration...")
    
    # Reset data first
    mujoco.mj_resetData(model, data)
    
    # Get home positions
    UR_HOME, HAND_HOME = get_home_positions()
    
    # Apply joint positions
    apply_joint_positions(UR_HOME, HAND_HOME, model, data)
    
    # Initialize box in safe position
    initialize_box_position(model, data)
    
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
    settle_robot(model, data, UR_HOME, steps=2000)
    
    print("Robot settled. Final joint positions:")
    for jn in UR_HOME.keys():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            current_pos = data.qpos[model.jnt_qposadr[jid]]
            target_pos = UR_HOME[jn]
            print(f"  {jn}: target={target_pos:.3f}, current={current_pos:.3f}, error={abs(current_pos-target_pos):.4f}")
    
    # Place the box correctly
    place_box_correctly(model, data)
    
    # Store the desired configuration
    desired_qpos = data.qpos.copy()
    desired_qvel = data.qvel.copy()
    
    print("Robot configuration setup complete!")
    return UR_HOME, HAND_HOME

def reset_to_desired_state(model, data):
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
    apply_hold_control(model, data, act_joint_qadr, ctrl_hold)
    
    # Forward kinematics to update all positions
    mujoco.mj_forward(model, data)
    
    reset_flag = False
    print("Reset complete")

def check_if_reset_needed(model, data):
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

def settle_robot(model, data, UR_HOME, steps=2000):
    """Settle the robot into position with control"""
    print("Settling robot into position...")
    for step in range(steps):
        # Apply control
        apply_hold_control(model, data, act_joint_qadr, ctrl_hold)
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

def place_box_correctly(model, data):
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
                apply_hold_control(model, data, act_joint_qadr, ctrl_hold)
                mujoco.mj_step(model, data)
            
            print(f"Box final position: {data.qpos[box_qpos_addr:box_qpos_addr+3]}")
    else:
        print(f"[ERROR] Site '{WRIST_SITE}' not found!")
