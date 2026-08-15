"""Small helpers for composing MuJoCo MJCF XML."""
from __future__ import annotations


def mjcf_header(timestep: float = 0.002, gravity: str = "0 0 -9.81") -> str:
    """Return a standard MuJoCo XML header."""
    return f"""<mujoco model="pibench_scene">
  <compiler angle="radian" />
  <option timestep="{timestep}" gravity="{gravity}" iterations="50" solver="Newton" />
  <visual>
    <global azimuth="120" elevation="-20" />
  </visual>
"""


def mjcf_worldbody_floor(size: float = 5.0) -> str:
    """Return a worldbody with a light and a floor."""
    return f"""
  <worldbody>
    <light directional="true" diffuse=".5 .5 .5" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1" />
    <light directional="false" diffuse=".6 .6 .6" specular=".3 .3 .3" pos="2 2 4" dir="-2 -2 -4" />
    <geom name="floor" type="plane" size="{size} {size} 0.1" rgba=".3 .3 .3 1" />
"""


def mjcf_box(name: str, pos: tuple[float, float, float], size: tuple[float, float, float],
             rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
             mass: float | None = None, friction: tuple[float, float, float] | None = None) -> str:
    """Return a free-floating box body."""
    friction_attr = ""
    if friction is not None:
        friction_attr = f' friction="{friction[0]} {friction[1]} {friction[2]}"'
    inertial = ""
    if mass is not None:
        inertial = f'\n      <inertial pos="0 0 0" mass="{mass}" diaginertia="{mass/12*(size[1]**2+size[2]**2)} {mass/12*(size[0]**2+size[2]**2)} {mass/12*(size[0]**2+size[1]**2)}" />'
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">{inertial}
      <freejoint />
      <geom name="{name}_geom" type="box" size="{size[0]} {size[1]} {size[2]}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"{friction_attr} />
    </body>
"""


def mjcf_incline(name: str, pos: tuple[float, float, float], size: tuple[float, float, float],
                 angle: float, rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
                 friction: tuple[float, float, float] | None = None) -> str:
    """Return a static inclined plane (box rotated about Y by angle).

    The box is sized as half-size (length, width, thickness). A positive angle
    tilts the high end toward +X.
    """
    friction_attr = ""
    if friction is not None:
        friction_attr = f' friction="{friction[0]} {friction[1]} {friction[2]}"'
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" euler="0 {angle} 0">
      <geom name="{name}_geom" type="box" size="{size[0]} {size[1]} {size[2]}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"{friction_attr} />
    </body>
"""


def mjcf_sphere(name: str, pos: tuple[float, float, float], radius: float,
                rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
                mass: float | None = None, friction: tuple[float, float, float] | None = None) -> str:
    """Return a free-floating sphere body."""
    friction_attr = ""
    if friction is not None:
        friction_attr = f' friction="{friction[0]} {friction[1]} {friction[2]}"'
    inertial = ""
    if mass is not None:
        # Solid sphere inertia: (2/5) * m * r^2.
        i = 0.4 * mass * radius * radius
        inertial = f'\n      <inertial pos="0 0 0" mass="{mass}" diaginertia="{i} {i} {i}" />'
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">{inertial}
      <freejoint />
      <geom name="{name}_geom" type="sphere" size="{radius}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"{friction_attr} />
    </body>
"""


def mjcf_footer() -> str:
    return "\n  </worldbody>\n</mujoco>\n"


def build_xml(worldbody_inner: str, timestep: float = 0.002, gravity: str = "0 0 -9.81") -> str:
    """Assemble a complete MJCF XML from the inner worldbody content."""
    return mjcf_header(timestep, gravity) + mjcf_worldbody_floor() + worldbody_inner + mjcf_footer()
