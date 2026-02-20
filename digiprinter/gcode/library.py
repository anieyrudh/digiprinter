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
