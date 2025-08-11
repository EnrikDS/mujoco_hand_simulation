import mujoco, numpy as np
from pathlib import Path

XML = Path(__file__).resolve().parents[1] / "mj" / "full_combined.xml"

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def axis_angle_to_quat(axis, angle):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    s = np.sin(angle/2.0)
    return np.array([np.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s])

# --- tweak these until fingertips face the right way and nothing intersects ---
# start with 180° about X if the hand is upside-down; otherwise leave 0.0
ROT_OFFSET = axis_angle_to_quat([1,0,0], 0.0)   # try np.pi if needed

# translation offset in the wrist-frame, meters.
# Positive Z usually "out of the flange". Try a few cm outwards.
TRANS_OFFSET = np.array([0.0, 0.0, 0.06])       # adjust while tuning
# -----------------------------------------------------------------------------

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
if sid < 0:
    raise RuntimeError("Site 'attachment_site' not found")

pos_w = d.site_xpos[sid].copy()
R_w   = d.site_xmat[sid].reshape(3,3).copy()
quat_w = np.empty(4); mujoco.mju_mat2Quat(quat_w, R_w.flatten(order='C'))

# final palm orientation in world
quat_palm = quat_mul(quat_w, ROT_OFFSET)
# rotate local offset by wrist orientation
pos_palm = pos_w + R_w @ TRANS_OFFSET

print("\n-- Paste into Allegro 'palm' body header --")
print(f'pos="{pos_palm[0]:.6f} {pos_palm[1]:.6f} {pos_palm[2]:.6f}"')
print(f'quat="{quat_palm[0]:.6f} {quat_palm[1]:.6f} {quat_palm[2]:.6f} {quat_palm[3]:.6f}"')

print("\n-- Weld to paste in <equality> (top-level) --")
print('<weld body1="wrist_3_link" body2="palm" solimp="0.99 0.99 0.01" solref="0.0005 1"/>')
