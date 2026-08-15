"""
Chapter 2 — Practical: Rigid-Body Motions.

Demonstrates:
  * Rotation matrices (SO(3))
  * Euler angles -> rotation matrix and back
  * Axis-angle (Rodrigues) rotation
  * Homogeneous transforms (SE(3))
  * Composition of transforms = post-multiplication
  * Transforming points between frames
"""
from __future__ import annotations

import math

import numpy as np


def rotx(theta: float) -> np.ndarray:
    """Return 3x3 rotation matrix for rotation about X-axis by theta radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def roty(theta: float) -> np.ndarray:
    """Return 3x3 rotation matrix for rotation about Y-axis by theta radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def rotz(theta: float) -> np.ndarray:
    """Return 3x3 rotation matrix for rotation about Z-axis by theta radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from XYZ intrinsic (fixed-axis) Euler angles.

    Order: rotate about X by roll, then Y by pitch, then Z by yaw.
    Intrinsic rotations are equivalent to multiplying the fixed-axis
    matrices in reverse order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Recover XYZ intrinsic Euler angles from a rotation matrix.

    euler_xyz builds R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (intrinsic ZYX,
    equivalent to extrinsic XYZ). The inverse decomposition is:
      pitch = asin(-R[2, 0])
      yaw   = atan2(R[1, 0], R[0, 0])
      roll  = atan2(R[2, 1], R[2, 2])
    """
    pitch = math.asin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = math.atan2(R[1, 0], R[0, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    return roll, pitch, yaw


def axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotation matrix for angle theta about unit axis."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    I = np.eye(3)
    return I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def homogeneous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation R and translation p."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def transform_point(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to a 3D point."""
    x_h = np.append(np.asarray(x, dtype=float), 1.0)
    y_h = T @ x_h
    return y_h[:3]


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) matrix: T^-1 = [R^T | -R^T p]."""
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def demo_rotation_matrices() -> None:
    print("=" * 70)
    print("DEMO: SO(3) rotation matrices")
    print("=" * 70)

    # A 90-degree rotation about Z sends the X axis to the Y axis.
    R_z90 = rotz(math.radians(90))
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = R_z90 @ x_axis
    print(f"Rz(90°) * [1,0,0] = {y_axis}")
    assert np.allclose(y_axis, [0.0, 1.0, 0.0])

    # Composition: Rz(90) then Ry(90)
    R = roty(math.radians(90)) @ rotz(math.radians(90))
    print(f"Ry(90°) @ Rz(90°) =\n{R}")

    # Rotation matrix properties.
    print(f"det(R) = {np.linalg.det(R):.6f}  (must be +1)")
    print(f"R @ R.T = identity? {np.allclose(R @ R.T, np.eye(3))}")
    print()


def demo_euler_angles() -> None:
    print("=" * 70)
    print("DEMO: Euler angles -> rotation matrix -> Euler angles")
    print("=" * 70)

    roll = math.radians(30)
    pitch = math.radians(45)
    yaw = math.radians(60)

    R = euler_xyz(roll, pitch, yaw)
    print(f"Input angles: roll={math.degrees(roll):.1f}°, pitch={math.degrees(pitch):.1f}°, yaw={math.degrees(yaw):.1f}°")
    print(f"R =\n{R}")

    recovered = rotation_matrix_to_euler_xyz(R)
    print(f"Recovered angles: roll={math.degrees(recovered[0]):.1f}°, "
          f"pitch={math.degrees(recovered[1]):.1f}°, yaw={math.degrees(recovered[2]):.1f}°")
    assert np.allclose(recovered, [roll, pitch, yaw])
    print()


def demo_axis_angle() -> None:
    print("=" * 70)
    print("DEMO: Axis-angle (Rodrigues) rotation")
    print("=" * 70)

    axis = np.array([0.0, 0.0, 1.0])
    theta = math.radians(90)
    R = axis_angle(axis, theta)
    print(f"Axis={axis}, angle={math.degrees(theta):.1f}°")
    print(f"R =\n{R}")
    v = np.array([1.0, 0.0, 0.0])
    print(f"R * {v} = {R @ v}")
    assert np.allclose(R @ v, [0.0, 1.0, 0.0])
    print()


def demo_homogeneous_transforms() -> None:
    print("=" * 70)
    print("DEMO: SE(3) homogeneous transforms and frame composition")
    print("=" * 70)

    # Frame {A} is rotated 90° about Z and translated by (1, 2, 0) in world frame.
    T_wa = homogeneous_transform(rotz(math.radians(90)), [1.0, 2.0, 0.0])

    # Frame {B} is rotated 90° about Y and translated by (0.5, 0, 0) in frame {A}.
    T_ab = homogeneous_transform(roty(math.radians(90)), [0.5, 0.0, 0.0])

    # World-to-B is the product T_wb = T_wa @ T_ab (read right-to-left).
    T_wb = T_wa @ T_ab
    print(f"T_wa =\n{T_wa}")
    print(f"T_ab =\n{T_ab}")
    print(f"T_wb = T_wa @ T_ab =\n{T_wb}")

    # A point in frame B: (0.5, 0, 0).
    p_b = np.array([0.5, 0.0, 0.0])
    p_w = transform_point(T_wb, p_b)
    print(f"Point p in frame B: {p_b}")
    print(f"Point p in world frame: {p_w}")

    # Inverse: express world point back in frame A.
    T_aw = inverse_transform(T_wa)
    p_a = transform_point(T_aw, p_w)
    print(f"World point in frame A: {p_a}")

    # Sanity check: T_wa * T_aw = identity.
    print(f"T_wa @ T_aw = identity? {np.allclose(T_wa @ T_aw, np.eye(4))}")
    print()


def main() -> None:
    demo_rotation_matrices()
    demo_euler_angles()
    demo_axis_angle()
    demo_homogeneous_transforms()

    print("=" * 70)
    print("Chapter 2 practical complete.")
    print("Key takeaway: rigid-body motion = rotation in SO(3) + translation,")
    print("composed via 4x4 homogeneous transforms in SE(3).")
    print("=" * 70)


if __name__ == "__main__":
    main()
