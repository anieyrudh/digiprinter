"""Matplotlib-based real-time dashboard renderer for the Prusa Core One+ environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from digiprinter.printer.state import PrinterState


class PrinterRenderer:
    """Renders a live 2x2 matplotlib dashboard during simulation.

    Layout
    ------
    Top-left : Toolpath visualisation (XY, extruded segments coloured by
               speed, travel moves as dashed gray lines).
    Top-right : Temperature traces over time (hotend, bed, chamber actual
                values as solid lines; hotend/bed targets as dashed).
    Bottom-left : Quality-metrics bar chart (adhesion, warping, stringing,
                  dimensional accuracy) updated every frame.
    Bottom-right : Status text panel (layer, G-code line, speed, flow,
                   fan %, reward).
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(self, figsize: tuple[float, float] = (14, 10)) -> None:
        import matplotlib
        matplotlib.use("TkAgg")          # interactive back-end
        import matplotlib.pyplot as plt

        self._plt = plt
        self._plt.ion()                   # interactive mode

        self._fig, self._axes = plt.subplots(2, 2, figsize=figsize)
        self._fig.suptitle("Prusa Core One+ -- Live Dashboard", fontsize=13)
        self._fig.tight_layout(rect=[0, 0, 1, 0.96])

        self._ax_toolpath: plt.Axes = self._axes[0, 0]
        self._ax_temp: plt.Axes = self._axes[0, 1]
        self._ax_quality: plt.Axes = self._axes[1, 0]
        self._ax_status: plt.Axes = self._axes[1, 1]

        # ---- History buffers -----------------------------------------
        # Toolpath: list of (x0, y0, x1, y1, speed, is_travel)
        self._toolpath_segments: list[tuple[float, float, float, float, float, bool]] = []
        self._prev_x: float | None = None
        self._prev_y: float | None = None

        # Temperature history: lists of floats
        self._time_hist: list[float] = []
        self._hotend_hist: list[float] = []
        self._bed_hist: list[float] = []
        self._chamber_hist: list[float] = []
        self._hotend_target_hist: list[float] = []
        self._bed_target_hist: list[float] = []

        # Quality (latest snapshot)
        self._quality_labels = ["Adhesion", "Warping\n(inv.)", "Stringing\n(inv.)", "Dim.\nAccuracy"]
        self._quality_values = [1.0, 1.0, 1.0, 1.0]

        # ---- Pre-configure static subplot properties -----------------
        self._setup_toolpath_axis()
        self._setup_temp_axis()
        self._setup_quality_axis()
        self._setup_status_axis()

        # Store line artists for temperature (faster updates)
        (self._line_hotend,) = self._ax_temp.plot([], [], "r-", lw=1.2, label="Hotend")
        (self._line_bed,) = self._ax_temp.plot([], [], "b-", lw=1.2, label="Bed")
        (self._line_chamber,) = self._ax_temp.plot([], [], "g-", lw=1.2, label="Chamber")
        (self._line_hotend_tgt,) = self._ax_temp.plot([], [], "r--", lw=0.8, alpha=0.6, label="Hotend tgt")
        (self._line_bed_tgt,) = self._ax_temp.plot([], [], "b--", lw=0.8, alpha=0.6, label="Bed tgt")
        self._ax_temp.legend(loc="upper left", fontsize=7)

        # Colorbar for toolpath speed (added on first segment)
        self._sm: object | None = None
        self._cbar: object | None = None

        self._fig.canvas.draw()
        self._plt.pause(0.001)

    # ------------------------------------------------------------------
    #  Axis setup helpers
    # ------------------------------------------------------------------

    def _setup_toolpath_axis(self) -> None:
        ax = self._ax_toolpath
        ax.set_title("Toolpath (XY)", fontsize=10)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    def _setup_temp_axis(self) -> None:
        ax = self._ax_temp
        ax.set_title("Temperatures", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Temperature (\u00b0C)", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _setup_quality_axis(self) -> None:
        ax = self._ax_quality
        ax.set_title("Quality Metrics", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score (0\u20131)", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    def _setup_status_axis(self) -> None:
        ax = self._ax_status
        ax.set_title("Status", fontsize=10)
        ax.axis("off")
        self._status_text = ax.text(
            0.05, 0.95, "", transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(self, state: PrinterState, info: dict) -> None:
        """Append *state* to history and redraw all four subplots."""
        self._record_state(state, info)
        self._draw_toolpath(state)
        self._draw_temperatures()
        self._draw_quality()
        self._draw_status(state, info)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def reset(self) -> None:
        """Clear all history buffers and subplot contents."""
        self._toolpath_segments.clear()
        self._prev_x = None
        self._prev_y = None

        self._time_hist.clear()
        self._hotend_hist.clear()
        self._bed_hist.clear()
        self._chamber_hist.clear()
        self._hotend_target_hist.clear()
        self._bed_target_hist.clear()

        self._quality_values = [1.0, 1.0, 1.0, 1.0]

        # Clear axes and re-setup
        for ax in self._axes.flat:
            ax.cla()

        self._setup_toolpath_axis()
        self._setup_temp_axis()
        self._setup_quality_axis()
        self._setup_status_axis()

        # Re-create temperature line artists
        (self._line_hotend,) = self._ax_temp.plot([], [], "r-", lw=1.2, label="Hotend")
        (self._line_bed,) = self._ax_temp.plot([], [], "b-", lw=1.2, label="Bed")
        (self._line_chamber,) = self._ax_temp.plot([], [], "g-", lw=1.2, label="Chamber")
        (self._line_hotend_tgt,) = self._ax_temp.plot([], [], "r--", lw=0.8, alpha=0.6, label="Hotend tgt")
        (self._line_bed_tgt,) = self._ax_temp.plot([], [], "b--", lw=0.8, alpha=0.6, label="Bed tgt")
        self._ax_temp.legend(loc="upper left", fontsize=7)

        self._sm = None
        self._cbar = None

        self._fig.canvas.draw()
        self._plt.pause(0.001)

    def close(self) -> None:
        """Close the matplotlib figure and turn off interactive mode."""
        self._plt.ioff()
        self._plt.close(self._fig)

    def get_rgb_array(self) -> np.ndarray:
        """Return the current figure as an RGB numpy array (H, W, 3)."""
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        # RGBA -> RGB
        return arr[:, :, :3].copy()

    # ------------------------------------------------------------------
    #  Internal drawing helpers
    # ------------------------------------------------------------------

    def _record_state(self, state: PrinterState, info: dict) -> None:
        """Append the current state snapshot to all history buffers."""
        # Toolpath segment
        x, y = state.x, state.y
        if self._prev_x is not None:
            is_travel = (state.flow_rate == 0.0) or state.retracted
            self._toolpath_segments.append(
                (self._prev_x, self._prev_y, x, y, state.current_speed, is_travel)
            )
        self._prev_x = x
        self._prev_y = y

        # Temperature
        self._time_hist.append(state.sim_time)
        self._hotend_hist.append(state.hotend_temp)
        self._bed_hist.append(state.bed_temp)
        self._chamber_hist.append(state.chamber_temp)
        self._hotend_target_hist.append(state.hotend_target)
        self._bed_target_hist.append(state.bed_target)

        # Quality metrics (invert warping/stringing so 1 = good, 0 = bad)
        max_warp = 2.0   # normalisation ceiling for warping (mm)
        max_string = 20.0  # normalisation ceiling for stringing (mm)
        self._quality_values = [
            float(np.clip(state.adhesion_quality, 0.0, 1.0)),
            float(np.clip(1.0 - state.warping_amount / max_warp, 0.0, 1.0)),
            float(np.clip(1.0 - state.stringing_amount / max_string, 0.0, 1.0)),
            float(np.clip(1.0 - state.dimensional_error, 0.0, 1.0)),
        ]

    def _draw_toolpath(self, state: PrinterState) -> None:
        """Draw the latest toolpath segment on the toolpath axis."""
        ax = self._ax_toolpath

        if len(self._toolpath_segments) == 0:
            return

        # Only draw the newest segment for performance
        seg = self._toolpath_segments[-1]
        x0, y0, x1, y1, speed, is_travel = seg

        if is_travel:
            ax.plot([x0, x1], [y0, y1], color="gray", ls="--", lw=0.4, alpha=0.4)
        else:
            # Map speed to colour: 0 mm/s -> blue, 200 mm/s -> red
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm

            if self._sm is None:
                self._sm = cm.ScalarMappable(
                    cmap=cm.get_cmap("plasma"),
                    norm=mcolors.Normalize(vmin=0, vmax=200),
                )
                self._sm.set_array([])
                self._cbar = self._fig.colorbar(
                    self._sm, ax=ax, pad=0.02, fraction=0.046, label="Speed (mm/s)"
                )

            colour = self._sm.to_rgba(speed)
            ax.plot([x0, x1], [y0, y1], color=colour, lw=0.7)

    def _draw_temperatures(self) -> None:
        """Update the temperature line artists with current history."""
        t = self._time_hist
        self._line_hotend.set_data(t, self._hotend_hist)
        self._line_bed.set_data(t, self._bed_hist)
        self._line_chamber.set_data(t, self._chamber_hist)
        self._line_hotend_tgt.set_data(t, self._hotend_target_hist)
        self._line_bed_tgt.set_data(t, self._bed_target_hist)

        ax = self._ax_temp
        ax.relim()
        ax.autoscale_view()

    def _draw_quality(self) -> None:
        """Redraw the quality-metrics bar chart."""
        ax = self._ax_quality
        ax.cla()
        self._setup_quality_axis()

        colours = ["#2ecc71", "#e67e22", "#e74c3c", "#3498db"]
        bars = ax.bar(
            self._quality_labels,
            self._quality_values,
            color=colours,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top of bars
        for bar, val in zip(bars, self._quality_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    def _draw_status(self, state: PrinterState, info: dict) -> None:
        """Update the status text panel."""
        reward = info.get("reward_components", {})
        reward_total = sum(reward.values()) if isinstance(reward, dict) else 0.0

        progress = info.get("progress", 0.0)
        layer_str = f"{state.current_layer}/{state.total_layers}" if state.total_layers else str(state.current_layer)
        gcode_str = f"{state.gcode_line}/{state.total_gcode_lines}" if state.total_gcode_lines else str(state.gcode_line)

        lines = [
            f"Layer:       {layer_str}",
            f"G-code line: {gcode_str}",
            f"Progress:    {progress:.1%}",
            "",
            f"Speed:       {state.current_speed:>7.1f} mm/s",
            f"Flow rate:   {state.flow_rate:>7.2f} mm\u00b3/s",
            f"Fan:         {state.fan_speed:>7.0%}",
            "",
            f"Hotend:      {state.hotend_temp:>6.1f} / {state.hotend_target:.0f} \u00b0C",
            f"Bed:         {state.bed_temp:>6.1f} / {state.bed_target:.0f} \u00b0C",
            f"Chamber:     {state.chamber_temp:>6.1f} \u00b0C",
            "",
            f"Reward:      {reward_total:>+7.3f}",
            f"Fault:       {state.fault or 'none'}",
        ]
        self._status_text.set_text("\n".join(lines))
