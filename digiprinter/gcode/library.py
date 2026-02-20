"""Benchmark G-code sequences.

Provides functions that return ready-to-use G-code strings for common
calibration and testing prints. Useful for quick simulation benchmarks
without requiring external .gcode files.
"""

from __future__ import annotations

import math


# Extrusion constant: filament cross-section area for 1.75 mm filament
_FILAMENT_RADIUS = 1.75 / 2.0  # 0.875 mm
_FILAMENT_AREA = math.pi * _FILAMENT_RADIUS ** 2  # ~2.405 mm^2


def _extrusion_length(
    segment_length: float,
    layer_height: float,
    extrusion_width: float = 0.4,
) -> float:
    """Calculate filament length to extrude for a given segment.

    Uses volumetric equivalence:
        volume_deposited = segment_length * layer_height * extrusion_width
        E = volume_deposited / filament_cross_section_area

    Parameters
    ----------
    segment_length:
        Length of the printed segment in mm.
    layer_height:
        Layer height in mm.
    extrusion_width:
        Extrusion width in mm (default 0.4 for a 0.4 mm nozzle).

    Returns
    -------
    Filament length to extrude in mm.
    """
    volume = segment_length * layer_height * extrusion_width
    return volume / _FILAMENT_AREA


def calibration_cube_gcode(
    size_mm: float = 20.0,
    layer_height: float = 0.2,
    nozzle_temp: float = 210.0,
    bed_temp: float = 60.0,
    print_speed: float = 60.0,
) -> str:
    """Generate G-code for a simple calibration cube.

    Produces a hollow cube (perimeter only) with the given dimensions.
    Useful as a quick benchmark for the simulation engine.

    Parameters
    ----------
    size_mm:
        Side length of the cube in mm.
    layer_height:
        Layer height in mm.
    nozzle_temp:
        Hotend temperature in degrees C.
    bed_temp:
        Bed temperature in degrees C.
    print_speed:
        Print speed in mm/s (converted to F in mm/min for G-code).

    Returns
    -------
    Multi-line G-code string.
    """
    feedrate = print_speed * 60.0  # mm/s -> mm/min
    travel_feedrate = 120.0 * 60.0  # fast travel
    z_feedrate = 300.0  # mm/min for Z moves

    num_layers = int(size_mm / layer_height)
    # Cap at 20 layers for a quick benchmark
    num_layers = min(num_layers, 20)

    # Cube origin offset (centered roughly on bed)
    ox, oy = 100.0, 100.0

    lines: list[str] = []

    # -- Preamble --
    lines.append("; Calibration cube")
    lines.append(f"; Size: {size_mm} mm, Layers: {num_layers}, Layer height: {layer_height} mm")
    lines.append(f"M104 S{nozzle_temp:.0f} ; set hotend temp")
    lines.append(f"M140 S{bed_temp:.0f} ; set bed temp")
    lines.append(f"M109 S{nozzle_temp:.0f} ; wait for hotend")
    lines.append(f"M190 S{bed_temp:.0f} ; wait for bed")
    lines.append("G28 ; home all axes")
    lines.append("M82 ; absolute extrusion")
    lines.append("G90 ; absolute positioning")
    lines.append("M106 S255 ; fan on full")
    lines.append("")

    e_total = 0.0  # running extrusion counter

    for layer in range(num_layers):
        z = (layer + 1) * layer_height
        lines.append(f"; Layer {layer}")
        lines.append(f"G1 Z{z:.3f} F{z_feedrate:.0f}")

        # Travel to start corner
        x0, y0 = ox, oy
        x1, y1 = ox + size_mm, oy
        x2, y2 = ox + size_mm, oy + size_mm
        x3, y3 = ox, oy + size_mm

        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{travel_feedrate:.0f}")

        # Four perimeter segments forming the square
        segments = [
            (x1, y1),  # bottom edge
            (x2, y2),  # right edge
            (x3, y3),  # top edge
            (x0, y0),  # left edge (back to start)
        ]

        for sx, sy in segments:
            seg_len = size_mm  # each side is size_mm
            e_total += _extrusion_length(seg_len, layer_height)
            lines.append(
                f"G1 X{sx:.3f} Y{sy:.3f} E{e_total:.5f} F{feedrate:.0f}"
            )

        lines.append("")

    # -- End G-code --
    lines.append("; End")
    lines.append("M104 S0 ; hotend off")
    lines.append("M140 S0 ; bed off")
    lines.append("M107 ; fan off")
    lines.append("G28 X Y ; home X Y")
    lines.append("M84 ; disable steppers")

    return "\n".join(lines) + "\n"


def single_line_gcode(
    length_mm: float = 100.0,
    layer_height: float = 0.2,
    nozzle_temp: float = 210.0,
    bed_temp: float = 60.0,
    print_speed: float = 30.0,
) -> str:
    """Generate G-code for a single straight line.

    Useful for the simplest possible extrusion test.

    Parameters
    ----------
    length_mm:
        Length of the line in mm.
    layer_height:
        Layer height in mm.
    nozzle_temp:
        Hotend temperature in degrees C.
    bed_temp:
        Bed temperature in degrees C.
    print_speed:
        Print speed in mm/s.

    Returns
    -------
    Multi-line G-code string.
    """
    feedrate = print_speed * 60.0
    travel_feedrate = 120.0 * 60.0

    start_x, start_y = 50.0, 100.0
    end_x = start_x + length_mm
    end_y = start_y

    e_length = _extrusion_length(length_mm, layer_height)

    lines: list[str] = [
        "; Single line test",
        f"M104 S{nozzle_temp:.0f}",
        f"M140 S{bed_temp:.0f}",
        f"M109 S{nozzle_temp:.0f}",
        f"M190 S{bed_temp:.0f}",
        "G28",
        "M82",
        "G90",
        "M106 S255",
        "",
        f"G1 Z{layer_height:.3f} F300",
        f"G0 X{start_x:.3f} Y{start_y:.3f} F{travel_feedrate:.0f}",
        f"G1 X{end_x:.3f} Y{end_y:.3f} E{e_length:.5f} F{feedrate:.0f}",
        "",
        "; End",
        "M104 S0",
        "M140 S0",
        "M107",
        "G28 X Y",
        "M84",
    ]

    return "\n".join(lines) + "\n"


def temperature_tower_gcode(
    start_temp: float = 230.0,
    end_temp: float = 190.0,
    temp_step: float = -5.0,
    section_height_mm: float = 5.0,
    section_size_mm: float = 15.0,
    layer_height: float = 0.2,
    bed_temp: float = 60.0,
    print_speed: float = 40.0,
) -> str:
    """Generate G-code for a temperature tower.

    Prints rectangular sections at progressively different temperatures.
    Each section is a small hollow rectangle repeated for
    ``section_height_mm / layer_height`` layers.

    Parameters
    ----------
    start_temp:
        Temperature for the first (bottom) section in degrees C.
    end_temp:
        Temperature for the last (top) section in degrees C.
    temp_step:
        Temperature change per section (negative for cooling tower).
    section_height_mm:
        Height of each temperature section in mm.
    section_size_mm:
        Width / depth of the printed rectangle in mm.
    layer_height:
        Layer height in mm.
    bed_temp:
        Bed temperature in degrees C.
    print_speed:
        Print speed in mm/s.

    Returns
    -------
    Multi-line G-code string.
    """
    feedrate = print_speed * 60.0
    travel_feedrate = 120.0 * 60.0
    z_feedrate = 300.0

    # Build list of temperatures
    temperatures: list[float] = []
    temp = start_temp
    if temp_step < 0:
        while temp >= end_temp:
            temperatures.append(temp)
            temp += temp_step
    else:
        while temp <= end_temp:
            temperatures.append(temp)
            temp += temp_step

    if not temperatures:
        temperatures = [start_temp]

    layers_per_section = max(1, int(section_height_mm / layer_height))

    ox, oy = 100.0, 100.0  # origin offset

    lines: list[str] = []

    # -- Preamble --
    lines.append("; Temperature tower")
    lines.append(f"; Temps: {temperatures}")
    lines.append(f"M140 S{bed_temp:.0f} ; set bed temp")
    lines.append(f"M104 S{start_temp:.0f} ; preheat hotend")
    lines.append(f"M190 S{bed_temp:.0f} ; wait for bed")
    lines.append(f"M109 S{start_temp:.0f} ; wait for hotend")
    lines.append("G28")
    lines.append("M82")
    lines.append("G90")
    lines.append("M106 S255")
    lines.append("")

    e_total = 0.0
    global_layer = 0

    for section_idx, nozzle_temp in enumerate(temperatures):
        lines.append(f"; === Section {section_idx}: {nozzle_temp:.0f} C ===")
        lines.append(f"M104 S{nozzle_temp:.0f}")

        # Wait for temperature on the first layer of each section
        if section_idx > 0:
            lines.append(f"M109 S{nozzle_temp:.0f}")

        for local_layer in range(layers_per_section):
            global_layer += 1
            z = global_layer * layer_height

            lines.append(f"; Layer {global_layer} (section {section_idx})")
            lines.append(f"G1 Z{z:.3f} F{z_feedrate:.0f}")

            # Perimeter corners
            x0, y0 = ox, oy
            x1, y1 = ox + section_size_mm, oy
            x2, y2 = ox + section_size_mm, oy + section_size_mm
            x3, y3 = ox, oy + section_size_mm

            # Travel to start
            lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{travel_feedrate:.0f}")

            # Four perimeter moves
            for sx, sy in [(x1, y1), (x2, y2), (x3, y3), (x0, y0)]:
                e_total += _extrusion_length(section_size_mm, layer_height)
                lines.append(
                    f"G1 X{sx:.3f} Y{sy:.3f} E{e_total:.5f} F{feedrate:.0f}"
                )

        lines.append("")

    # -- End G-code --
    lines.append("; End")
    lines.append("M104 S0")
    lines.append("M140 S0")
    lines.append("M107")
    lines.append("G28 X Y")
    lines.append("M84")

    return "\n".join(lines) + "\n"


def benchy_simplified_gcode(
    nozzle_temp: float = 215.0,
    bed_temp: float = 60.0,
    print_speed: float = 60.0,
    layer_height: float = 0.2,
) -> str:
    """Generate G-code for a simplified 3DBenchy-like shape.

    The print has three vertical zones:
      - Layers 0-15: hull perimeter (rounded rectangle ~30x15 mm).
      - Layers 16-25: hull + cabin (smaller rectangle ~10x8 mm on top).
      - Layers 26-30: hull + cabin + chimney (tiny 3x3 mm square).

    Approximately 30 layers total.  The hull "rounded rectangle" is
    approximated with semicircular end-caps (8 segments each) joined by
    straight sides.

    Parameters
    ----------
    nozzle_temp:
        Hotend temperature in degrees C.
    bed_temp:
        Bed temperature in degrees C.
    print_speed:
        Print speed in mm/s.
    layer_height:
        Layer height in mm.

    Returns
    -------
    Multi-line G-code string.
    """
    feedrate = print_speed * 60.0
    travel_feedrate = 120.0 * 60.0
    z_feedrate = 300.0

    num_layers = 30

    # Origins (roughly centred on a 200 mm bed)
    ox, oy = 85.0, 92.5

    # Hull dimensions
    hull_w, hull_h = 30.0, 15.0
    hull_r = hull_h / 2.0  # semicircle radius for end-caps
    arc_segments = 8  # segments per semicircle

    # Cabin rectangle (offset towards the "stern")
    cabin_w, cabin_h = 10.0, 8.0
    cabin_ox = ox + hull_w - cabin_w - 2.0
    cabin_oy = oy + (hull_h - cabin_h) / 2.0

    # Chimney square (small, centred on cabin)
    chim_s = 3.0
    chim_ox = cabin_ox + (cabin_w - chim_s) / 2.0
    chim_oy = cabin_oy + (cabin_h - chim_s) / 2.0

    lines: list[str] = []

    # -- Preamble --
    lines.append("; Simplified 3DBenchy")
    lines.append(f"; Layers: {num_layers}, Layer height: {layer_height} mm")
    lines.append(f"M104 S{nozzle_temp:.0f} ; set hotend temp")
    lines.append(f"M140 S{bed_temp:.0f} ; set bed temp")
    lines.append(f"M109 S{nozzle_temp:.0f} ; wait for hotend")
    lines.append(f"M190 S{bed_temp:.0f} ; wait for bed")
    lines.append("G28 ; home all axes")
    lines.append("M82 ; absolute extrusion")
    lines.append("G90 ; absolute positioning")
    lines.append("M106 S255 ; fan on full")
    lines.append("")

    # Pre-compute rounded-rectangle vertices for the hull
    def _hull_perimeter() -> list[tuple[float, float]]:
        """Return an ordered list of (x, y) points tracing the hull."""
        pts: list[tuple[float, float]] = []
        # Bottom straight edge (left to right)
        pts.append((ox + hull_r, oy))
        pts.append((ox + hull_w - hull_r, oy))
        # Right semicircle (bottom to top)
        cx_r = ox + hull_w - hull_r
        cy_r = oy + hull_r
        for i in range(1, arc_segments + 1):
            angle = -math.pi / 2.0 + math.pi * i / arc_segments
            pts.append((cx_r + hull_r * math.cos(angle),
                        cy_r + hull_r * math.sin(angle)))
        # Top straight edge (right to left)
        pts.append((ox + hull_r, oy + hull_h))
        # Left semicircle (top to bottom)
        cx_l = ox + hull_r
        cy_l = oy + hull_r
        for i in range(1, arc_segments + 1):
            angle = math.pi / 2.0 + math.pi * i / arc_segments
            pts.append((cx_l + hull_r * math.cos(angle),
                        cy_l + hull_r * math.sin(angle)))
        return pts

    hull_pts = _hull_perimeter()

    def _rect_points(rx: float, ry: float, rw: float, rh: float):
        return [
            (rx, ry),
            (rx + rw, ry),
            (rx + rw, ry + rh),
            (rx, ry + rh),
        ]

    cabin_pts = _rect_points(cabin_ox, cabin_oy, cabin_w, cabin_h)
    chimney_pts = _rect_points(chim_ox, chim_oy, chim_s, chim_s)

    e_total = 0.0

    def _emit_perimeter(pts: list[tuple[float, float]]) -> None:
        nonlocal e_total
        # Travel to first point
        lines.append(
            f"G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f} F{travel_feedrate:.0f}"
        )
        prev = pts[0]
        for pt in pts[1:]:
            seg_len = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            e_total += _extrusion_length(seg_len, layer_height)
            lines.append(
                f"G1 X{pt[0]:.3f} Y{pt[1]:.3f} E{e_total:.5f} F{feedrate:.0f}"
            )
            prev = pt
        # Close loop back to first point
        seg_len = math.hypot(pts[0][0] - prev[0], pts[0][1] - prev[1])
        e_total += _extrusion_length(seg_len, layer_height)
        lines.append(
            f"G1 X{pts[0][0]:.3f} Y{pts[0][1]:.3f} E{e_total:.5f} F{feedrate:.0f}"
        )

    for layer in range(num_layers):
        z = (layer + 1) * layer_height
        lines.append(f"; Layer {layer}")
        lines.append(f"G1 Z{z:.3f} F{z_feedrate:.0f}")

        # Hull (all layers)
        _emit_perimeter(hull_pts)

        # Cabin (layers 16-30)
        if layer >= 16:
            _emit_perimeter(cabin_pts)

        # Chimney (layers 26-30)
        if layer >= 26:
            _emit_perimeter(chimney_pts)

        lines.append("")

    # -- End G-code --
    lines.append("; End")
    lines.append("M104 S0 ; hotend off")
    lines.append("M140 S0 ; bed off")
    lines.append("M107 ; fan off")
    lines.append("G28 X Y ; home X Y")
    lines.append("M84 ; disable steppers")

    return "\n".join(lines) + "\n"


def overhang_test_gcode(
    nozzle_temp: float = 210.0,
    bed_temp: float = 60.0,
    print_speed: float = 40.0,
    layer_height: float = 0.2,
) -> str:
    """Generate G-code for a stepped overhang test structure.

    A 30x10 mm rectangular base is printed for the first 5 layers.
    Each subsequent 5-layer block extends 2 mm beyond the previous
    block on one side, creating progressively steeper overhangs
    (~15deg, 30deg, 45deg, 60deg).  25 layers total with 5 overhang
    steps.

    This is useful for training an RL agent to handle increasing
    overhang difficulty.

    Parameters
    ----------
    nozzle_temp:
        Hotend temperature in degrees C.
    bed_temp:
        Bed temperature in degrees C.
    print_speed:
        Print speed in mm/s.
    layer_height:
        Layer height in mm.

    Returns
    -------
    Multi-line G-code string.
    """
    feedrate = print_speed * 60.0
    travel_feedrate = 120.0 * 60.0
    z_feedrate = 300.0

    num_layers = 25
    layers_per_step = 5
    overhang_shift = 2.0  # mm extension per step

    base_w, base_h = 30.0, 10.0
    ox, oy = 85.0, 95.0  # origin offset

    lines: list[str] = []

    # -- Preamble --
    lines.append("; Overhang test")
    lines.append(f"; Layers: {num_layers}, Layer height: {layer_height} mm")
    lines.append(f"M104 S{nozzle_temp:.0f} ; set hotend temp")
    lines.append(f"M140 S{bed_temp:.0f} ; set bed temp")
    lines.append(f"M109 S{nozzle_temp:.0f} ; wait for hotend")
    lines.append(f"M190 S{bed_temp:.0f} ; wait for bed")
    lines.append("G28 ; home all axes")
    lines.append("M82 ; absolute extrusion")
    lines.append("G90 ; absolute positioning")
    lines.append("M106 S255 ; fan on full")
    lines.append("")

    e_total = 0.0

    for layer in range(num_layers):
        z = (layer + 1) * layer_height
        step = layer // layers_per_step  # 0-based step index
        extension = step * overhang_shift  # cumulative overhang

        # The rectangle grows wider on the +X side each step
        x0 = ox
        y0 = oy
        x1 = ox + base_w + extension
        y1 = oy + base_h

        w = x1 - x0
        h = y1 - y0

        lines.append(f"; Layer {layer} (step {step}, overhang +{extension:.1f} mm)")
        lines.append(f"G1 Z{z:.3f} F{z_feedrate:.0f}")

        # Travel to start corner
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{travel_feedrate:.0f}")

        # Perimeter: bottom -> right -> top -> left (back to start)
        segments = [
            ((x1, y0), w),
            ((x1, y1), h),
            ((x0, y1), w),
            ((x0, y0), h),
        ]

        for (sx, sy), seg_len in segments:
            e_total += _extrusion_length(seg_len, layer_height)
            lines.append(
                f"G1 X{sx:.3f} Y{sy:.3f} E{e_total:.5f} F{feedrate:.0f}"
            )

        lines.append("")

    # -- End G-code --
    lines.append("; End")
    lines.append("M104 S0 ; hotend off")
    lines.append("M140 S0 ; bed off")
    lines.append("M107 ; fan off")
    lines.append("G28 X Y ; home X Y")
    lines.append("M84 ; disable steppers")

    return "\n".join(lines) + "\n"


def bridge_test_gcode(
    nozzle_temp: float = 210.0,
    bed_temp: float = 60.0,
    print_speed: float = 30.0,
    layer_height: float = 0.2,
) -> str:
    """Generate G-code for a bridging test.

    Two 5x5 mm pillars separated by a 20 mm gap are printed for 10
    layers, then a bridging layer connects the tops of the two pillars.
    Tests the simulation's handling of unsupported (bridging) extrusion.

    Parameters
    ----------
    nozzle_temp:
        Hotend temperature in degrees C.
    bed_temp:
        Bed temperature in degrees C.
    print_speed:
        Print speed in mm/s.
    layer_height:
        Layer height in mm.

    Returns
    -------
    Multi-line G-code string.
    """
    feedrate = print_speed * 60.0
    travel_feedrate = 120.0 * 60.0
    z_feedrate = 300.0

    pillar_size = 5.0
    gap = 20.0
    pillar_layers = 10

    # Left pillar origin
    lp_ox, lp_oy = 90.0, 97.5
    # Right pillar origin (shifted by pillar_size + gap)
    rp_ox = lp_ox + pillar_size + gap
    rp_oy = lp_oy

    lines: list[str] = []

    # -- Preamble --
    lines.append("; Bridge test")
    lines.append(f"; Pillar layers: {pillar_layers}, Gap: {gap} mm")
    lines.append(f"M104 S{nozzle_temp:.0f} ; set hotend temp")
    lines.append(f"M140 S{bed_temp:.0f} ; set bed temp")
    lines.append(f"M109 S{nozzle_temp:.0f} ; wait for hotend")
    lines.append(f"M190 S{bed_temp:.0f} ; wait for bed")
    lines.append("G28 ; home all axes")
    lines.append("M82 ; absolute extrusion")
    lines.append("G90 ; absolute positioning")
    lines.append("M106 S255 ; fan on full")
    lines.append("")

    e_total = 0.0

    def _emit_rect(rx: float, ry: float, rw: float, rh: float) -> None:
        """Emit a rectangular perimeter."""
        nonlocal e_total
        x0, y0 = rx, ry
        x1, y1 = rx + rw, ry
        x2, y2 = rx + rw, ry + rh
        x3, y3 = rx, ry + rh

        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{travel_feedrate:.0f}")
        for (sx, sy), seg_len in [
            ((x1, y1), rw),
            ((x2, y2), rh),
            ((x3, y3), rw),
            ((x0, y0), rh),
        ]:
            e_total += _extrusion_length(seg_len, layer_height)
            lines.append(
                f"G1 X{sx:.3f} Y{sy:.3f} E{e_total:.5f} F{feedrate:.0f}"
            )

    # Print both pillars for pillar_layers
    for layer in range(pillar_layers):
        z = (layer + 1) * layer_height
        lines.append(f"; Layer {layer} (pillars)")
        lines.append(f"G1 Z{z:.3f} F{z_feedrate:.0f}")

        _emit_rect(lp_ox, lp_oy, pillar_size, pillar_size)
        _emit_rect(rp_ox, rp_oy, pillar_size, pillar_size)
        lines.append("")

    # Bridging layer: connect the two pillars across the gap
    bridge_layer = pillar_layers
    z = (bridge_layer + 1) * layer_height
    lines.append(f"; Layer {bridge_layer} (bridge)")
    lines.append(f"G1 Z{z:.3f} F{z_feedrate:.0f}")

    # Continue printing both pillars on this layer
    _emit_rect(lp_ox, lp_oy, pillar_size, pillar_size)
    _emit_rect(rp_ox, rp_oy, pillar_size, pillar_size)

    # Bridge lines connecting left pillar top to right pillar top.
    # We lay several parallel bridge lines across the gap at different Y
    # offsets within the pillar width.
    bridge_x_start = lp_ox + pillar_size
    bridge_x_end = rp_ox
    bridge_span = bridge_x_end - bridge_x_start
    num_bridge_lines = 5
    for i in range(num_bridge_lines):
        y = lp_oy + pillar_size * (i + 0.5) / num_bridge_lines
        lines.append(f"G0 X{bridge_x_start:.3f} Y{y:.3f} F{travel_feedrate:.0f}")
        e_total += _extrusion_length(bridge_span, layer_height)
        lines.append(
            f"G1 X{bridge_x_end:.3f} Y{y:.3f} E{e_total:.5f} F{feedrate:.0f}"
        )
    lines.append("")

    # -- End G-code --
    lines.append("; End")
    lines.append("M104 S0 ; hotend off")
    lines.append("M140 S0 ; bed off")
    lines.append("M107 ; fan off")
    lines.append("G28 X Y ; home X Y")
    lines.append("M84 ; disable steppers")

    return "\n".join(lines) + "\n"


def spiral_vase_gcode(
    diameter: float = 40.0,
    height: float = 30.0,
    nozzle_temp: float = 210.0,
    bed_temp: float = 60.0,
    print_speed: float = 40.0,
    layer_height: float = 0.2,
) -> str:
    """Generate G-code for a spiral vase (single-wall, continuous Z rise).

    The vase is a cylinder approximated by 64 line segments per
    revolution.  Z increases continuously so there are no discrete
    layer transitions, mimicking "vase mode" / "spiralize outer contour".

    Good for testing smooth motion planning and consistent extrusion.

    Parameters
    ----------
    diameter:
        Outer diameter of the vase in mm.
    height:
        Total height of the vase in mm.
    nozzle_temp:
        Hotend temperature in degrees C.
    bed_temp:
        Bed temperature in degrees C.
    print_speed:
        Print speed in mm/s.
    layer_height:
        Layer height in mm.

    Returns
    -------
    Multi-line G-code string.
    """
    feedrate = print_speed * 60.0
    travel_feedrate = 120.0 * 60.0

    radius = diameter / 2.0
    segments_per_rev = 64
    total_revolutions = height / layer_height
    total_segments = int(total_revolutions * segments_per_rev)

    # Centre of vase on the bed
    cx, cy = 100.0, 100.0

    lines: list[str] = []

    # -- Preamble --
    lines.append("; Spiral vase")
    lines.append(f"; Diameter: {diameter} mm, Height: {height} mm")
    lines.append(f"; Segments/rev: {segments_per_rev}, Total segments: {total_segments}")
    lines.append(f"M104 S{nozzle_temp:.0f} ; set hotend temp")
    lines.append(f"M140 S{bed_temp:.0f} ; set bed temp")
    lines.append(f"M109 S{nozzle_temp:.0f} ; wait for hotend")
    lines.append(f"M190 S{bed_temp:.0f} ; wait for bed")
    lines.append("G28 ; home all axes")
    lines.append("M82 ; absolute extrusion")
    lines.append("G90 ; absolute positioning")
    lines.append("M106 S255 ; fan on full")
    lines.append("")

    # Move to start position (angle = 0, Z = first layer height)
    start_x = cx + radius
    start_y = cy
    start_z = layer_height
    lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f} F{travel_feedrate:.0f}")
    lines.append(f"G1 Z{start_z:.3f} F300")

    e_total = 0.0
    # Segment arc length for extrusion calculation
    seg_angle = 2.0 * math.pi / segments_per_rev
    z_per_segment = height / total_segments

    prev_x, prev_y = start_x, start_y

    for seg in range(1, total_segments + 1):
        angle = seg * seg_angle
        z = start_z + seg * z_per_segment
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        seg_len = math.hypot(x - prev_x, y - prev_y)
        e_total += _extrusion_length(seg_len, layer_height)

        lines.append(
            f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} E{e_total:.5f} F{feedrate:.0f}"
        )

        prev_x, prev_y = x, y

    lines.append("")

    # -- End G-code --
    lines.append("; End")
    lines.append("M104 S0 ; hotend off")
    lines.append("M140 S0 ; bed off")
    lines.append("M107 ; fan off")
    lines.append("G28 X Y ; home X Y")
    lines.append("M84 ; disable steppers")

    return "\n".join(lines) + "\n"
