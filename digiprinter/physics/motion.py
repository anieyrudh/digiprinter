"""CoreXY kinematics, trapezoidal motion planner, and ZV input shaper."""

from __future__ import annotations

import math

import numpy as np

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG


class MotionModel:
    """CoreXY motion model with trapezoidal velocity planning and ZV input
    shaping for a 3D-printer simulation.

    All distances are in millimetres, speeds in mm/s, accelerations in
    mm/s**2, and time in seconds.
    """

    # ------------------------------------------------------------------ #
    #  Construction / reset                                               #
    # ------------------------------------------------------------------ #

    def __init__(self, config: PrinterConfig = DEFAULT_CONFIG) -> None:
        self.config: PrinterConfig = config

        # --- Cartesian position (mm) ---
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0

        # --- Current velocity (mm/s) ---
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.vz: float = 0.0

        # --- CoreXY motor positions (mm of belt travel) ---
        self.motor_a: float = 0.0
        self.motor_b: float = 0.0

        # --- Pre-compute ZV input-shaper coefficients ---
        zeta = config.damping_ratio
        f_res = config.resonance_freq

        K = math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))
        self._shaper_amplitudes: list[float] = [K / (1.0 + K), 1.0 / (1.0 + K)]
        # Delay of the second impulse in seconds
        self._shaper_delay_s: float = 0.5 / f_res

    def reset(self) -> None:
        """Zero all positions and velocities."""
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.motor_a = 0.0
        self.motor_b = 0.0

    # ------------------------------------------------------------------ #
    #  CoreXY kinematics                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def forward_kinematics(a: float, b: float) -> tuple[float, float]:
        """Convert CoreXY motor positions to Cartesian (X, Y).

        X = 0.5 * (A + B)
        Y = 0.5 * (A - B)
        """
        x = 0.5 * (a + b)
        y = 0.5 * (a - b)
        return x, y

    @staticmethod
    def inverse_kinematics(x: float, y: float) -> tuple[float, float]:
        """Convert Cartesian (X, Y) to CoreXY motor positions (A, B).

        A = X + Y
        B = X - Y
        """
        a = x + y
        b = x - y
        return a, b

    # ------------------------------------------------------------------ #
    #  Trapezoidal motion planner                                         #
    # ------------------------------------------------------------------ #

    def plan_move(
        self,
        x_target: float,
        y_target: float,
        z_target: float,
        feedrate: float,
        dt: float,
    ) -> list[tuple[float, float, float, float]]:
        """Plan a trapezoidal-velocity-profile move and return waypoints.

        Parameters
        ----------
        x_target, y_target, z_target : float
            Target position in mm.
        feedrate : float
            Requested feedrate in mm/s.
        dt : float
            Time-step interval in seconds for waypoint generation.

        Returns
        -------
        list[tuple[float, float, float, float]]
            Waypoints as (x, y, z, speed) tuples sampled at *dt* intervals.
        """
        cfg = self.config

        dx = x_target - self.x
        dy = y_target - self.y
        dz = z_target - self.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 1e-6:
            return []

        # Unit direction vector
        ux = dx / distance
        uy = dy / distance
        uz = dz / distance

        # Determine whether this is a pure-Z move
        xy_dist = math.sqrt(dx * dx + dy * dy)
        is_z_only = xy_dist < 1e-6

        max_speed = cfg.max_speed_z if is_z_only else cfg.max_speed_xy
        accel = cfg.max_accel_z if is_z_only else cfg.max_accel_xy

        # Clamp feedrate
        feedrate = min(feedrate, max_speed)

        # --- Trapezoidal profile computation ---
        t_accel = feedrate / accel
        accel_dist = 0.5 * accel * t_accel * t_accel

        if 2.0 * accel_dist > distance:
            # Triangular profile — cannot reach requested feedrate
            peak_speed = math.sqrt(accel * distance)
            t_accel = peak_speed / accel
            t_cruise = 0.0
            t_decel = t_accel
            accel_dist = 0.5 * distance
            cruise_dist = 0.0
        else:
            peak_speed = feedrate
            cruise_dist = distance - 2.0 * accel_dist
            t_cruise = cruise_dist / feedrate
            t_decel = t_accel

        total_time = t_accel + t_cruise + t_decel

        # --- Generate waypoints at dt intervals ---
        waypoints: list[tuple[float, float, float, float]] = []

        t = 0.0
        while t <= total_time:
            if t < t_accel:
                # Acceleration phase
                speed = accel * t
                s = 0.5 * accel * t * t
            elif t < t_accel + t_cruise:
                # Cruise phase
                t_in_cruise = t - t_accel
                speed = peak_speed
                s = accel_dist + peak_speed * t_in_cruise
            else:
                # Deceleration phase
                t_in_decel = t - t_accel - t_cruise
                speed = peak_speed - accel * t_in_decel
                s = (
                    accel_dist
                    + cruise_dist
                    + peak_speed * t_in_decel
                    - 0.5 * accel * t_in_decel * t_in_decel
                )

            # Clamp distance to avoid overshooting due to float precision
            s = min(s, distance)
            speed = max(speed, 0.0)

            wx = self.x + ux * s
            wy = self.y + uy * s
            wz = self.z + uz * s

            waypoints.append((wx, wy, wz, speed))

            t += dt

        # Ensure final waypoint is exactly at the target
        if waypoints:
            last = waypoints[-1]
            if (
                abs(last[0] - x_target) > 1e-9
                or abs(last[1] - y_target) > 1e-9
                or abs(last[2] - z_target) > 1e-9
            ):
                waypoints.append((x_target, y_target, z_target, 0.0))

        return waypoints

    # ------------------------------------------------------------------ #
    #  Execute move (plan + update state)                                 #
    # ------------------------------------------------------------------ #

    def execute_move(
        self,
        x_target: float,
        y_target: float,
        z_target: float,
        feedrate: float,
        dt: float,
    ) -> list[tuple[float, float, float, float]]:
        """Plan a move, update internal state, and return waypoints.

        Parameters
        ----------
        x_target, y_target, z_target : float
            Target position in mm.
        feedrate : float
            Requested feedrate in mm/s.
        dt : float
            Time-step interval in seconds.

        Returns
        -------
        list[tuple[float, float, float, float]]
            Waypoints as (x, y, z, speed) tuples.
        """
        waypoints = self.plan_move(x_target, y_target, z_target, feedrate, dt)

        # Update Cartesian position
        self.x = x_target
        self.y = y_target
        self.z = z_target

        # Update CoreXY motor positions
        self.motor_a, self.motor_b = self.inverse_kinematics(self.x, self.y)

        return waypoints

    # ------------------------------------------------------------------ #
    #  ZV input shaper                                                    #
    # ------------------------------------------------------------------ #

    def apply_input_shaper(
        self,
        waypoints: list[tuple[float, float, float, float]],
        dt: float,
    ) -> list[tuple[float, float, float, float]]:
        """Apply a ZV (Zero Vibration) input shaper to a waypoint list.

        The shaper convolves the position trajectory with two impulses
        whose amplitudes and delays were pre-computed in ``__init__``.

        Parameters
        ----------
        waypoints : list[tuple[float, float, float, float]]
            Input waypoints as (x, y, z, speed).
        dt : float
            Time-step interval used to convert the delay to an index offset.

        Returns
        -------
        list[tuple[float, float, float, float]]
            Shaped waypoints with the same length as the input.
        """
        if not waypoints:
            return []

        n = len(waypoints)
        amps = self._shaper_amplitudes
        delay_steps = int(round(self._shaper_delay_s / dt)) if dt > 0.0 else 0
        delays = [0, delay_steps]

        shaped: list[tuple[float, float, float, float]] = []

        for i in range(n):
            sx = 0.0
            sy = 0.0
            sz = 0.0
            s_speed = 0.0

            for amp, delay in zip(amps, delays):
                idx = max(i - delay, 0)
                wp = waypoints[idx]
                sx += amp * wp[0]
                sy += amp * wp[1]
                sz += amp * wp[2]
                s_speed += amp * wp[3]

            shaped.append((sx, sy, sz, s_speed))

        return shaped

    # ------------------------------------------------------------------ #
    #  Junction speed                                                     #
    # ------------------------------------------------------------------ #

    def junction_speed(
        self,
        v1_vec: np.ndarray,
        v2_vec: np.ndarray,
        junction_deviation: float,
    ) -> float:
        """Compute the maximum allowable speed at the junction of two
        consecutive move segments.

        Uses a cornering model based on junction deviation:

            v_junction = sqrt(deviation * accel * sin(theta/2)
                              / (1 - sin(theta/2)))

        The result is clamped to the minimum of the two segment speeds.

        Parameters
        ----------
        v1_vec : np.ndarray
            Velocity vector of the first segment (mm/s).
        v2_vec : np.ndarray
            Velocity vector of the second segment (mm/s).
        junction_deviation : float
            Junction deviation parameter (mm).

        Returns
        -------
        float
            Maximum junction speed in mm/s.
        """
        cfg = self.config
        accel = cfg.max_accel_xy

        speed1 = float(np.linalg.norm(v1_vec))
        speed2 = float(np.linalg.norm(v2_vec))

        if speed1 < 1e-9 or speed2 < 1e-9:
            return 0.0

        # Unit vectors
        u1 = v1_vec / speed1
        u2 = v2_vec / speed2

        # Cosine of the angle between the two direction vectors
        cos_theta = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        # Full angle between segments
        theta = math.acos(cos_theta)

        half_theta = 0.5 * theta
        sin_half = math.sin(half_theta)

        if sin_half > 1.0 - 1e-9:
            # Segments are nearly anti-parallel — full stop required
            return 0.0

        v_junction = math.sqrt(
            junction_deviation * accel * sin_half / (1.0 - sin_half)
        )

        # Clamp to the slower of the two segment speeds
        v_junction = min(v_junction, speed1, speed2)

        return v_junction
