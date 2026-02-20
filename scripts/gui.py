#!/usr/bin/env python3
"""GUI dashboard for the DigiPrinter digital twin.

Launches a tkinter window with live visualization of the Prusa Core One+
simulation running under a trained SAC agent, random policy, or zero-action
baseline.

Usage:
    python scripts/gui.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so digiprinter is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


# =========================================================================== #
#  Main GUI class                                                              #
# =========================================================================== #

class DigiPrinterGUI:
    """Self-contained tkinter dashboard for the digital twin."""

    # ------------------------------------------------------------------ #
    #  Colour palette (dark theme)                                        #
    # ------------------------------------------------------------------ #
    BG = "#1a1a2e"
    PANEL_BG = "#16213e"
    ACCENT = "#0f3460"
    TEXT = "#e0e0e0"
    TEXT_DIM = "#8899aa"
    HOTEND_COLOR = "#ff6b6b"
    BED_COLOR = "#4ecdc4"
    CHAMBER_COLOR = "#ffe66d"
    EXTRUDE_COLOR = "#00ff88"
    TRAVEL_COLOR = "#333344"
    QUALITY_GOOD = "#00ff88"
    QUALITY_WARN = "#ffaa44"
    QUALITY_BAD = "#ff4444"
    BTN_ACTIVE = "#1a5276"

    # Prusa Core One+ build volume (mm)
    BED_X = 250.0
    BED_Y = 210.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DigiPrinter \u2014 Prusa Core One+ Digital Twin")
        self.root.geometry("960x740")
        self.root.minsize(800, 600)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ----- Mutable state ------------------------------------------
        self.running = False
        self.policy: str = "sac"
        self.path_segments: list[tuple[float, float, float, float, bool]] = []
        self.step_count = 0
        self.total_steps = 0
        self.total_reward = 0.0
        self._closing = False

        # ----- Build the interface ------------------------------------
        self._build_ui()

    # ================================================================== #
    #  UI construction                                                    #
    # ================================================================== #

    def _build_ui(self) -> None:
        # ---------- Title bar -----------------------------------------
        title_frame = tk.Frame(self.root, bg=self.ACCENT)
        title_frame.pack(fill=tk.X)
        tk.Label(
            title_frame,
            text="  DigiPrinter \u2014 Prusa Core One+ Digital Twin",
            font=("Helvetica", 15, "bold"),
            fg=self.TEXT,
            bg=self.ACCENT,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=8)

        # ---------- Main content area (left + right) ------------------
        content = tk.Frame(self.root, bg=self.BG)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 4))

        # LEFT: Toolpath canvas
        left = tk.Frame(content, bg=self.PANEL_BG, relief=tk.RIDGE, bd=2)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        canvas_header = tk.Frame(left, bg=self.PANEL_BG)
        canvas_header.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(
            canvas_header,
            text="Toolpath (XY)",
            font=("Helvetica", 11, "bold"),
            fg=self.TEXT,
            bg=self.PANEL_BG,
        ).pack(side=tk.LEFT)
        self._canvas_legend = tk.Label(
            canvas_header,
            text="\u2500 extrude   \u2500 travel",
            font=("Courier", 8),
            fg=self.TEXT_DIM,
            bg=self.PANEL_BG,
        )
        self._canvas_legend.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(left, bg="#080818", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # RIGHT: Gauges panel (fixed width)
        right = tk.Frame(content, bg=self.PANEL_BG, relief=tk.RIDGE, bd=2, width=310)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right.pack_propagate(False)

        # -- Temperature section --
        self._section_label(right, "Temperatures")
        self.hotend_gauge = self._make_gauge(right, "Hotend", self.HOTEND_COLOR)
        self.bed_gauge = self._make_gauge(right, "Bed", self.BED_COLOR)
        self.chamber_gauge = self._make_gauge(right, "Chamber", self.CHAMBER_COLOR)

        self._separator(right)

        # -- Quality metrics section --
        self._section_label(right, "Quality Metrics")
        self.adhesion_gauge = self._make_gauge(right, "Adhesion", self.QUALITY_GOOD)
        self.warping_gauge = self._make_gauge(right, "Warping", self.QUALITY_BAD)
        self.stringing_gauge = self._make_gauge(right, "Stringing", self.QUALITY_BAD)
        self.dimerr_gauge = self._make_gauge(right, "Dim Error", self.QUALITY_WARN)

        self._separator(right)

        # -- Reward display --
        self.reward_label = tk.Label(
            right,
            text="Reward: 0.00",
            font=("Courier", 16, "bold"),
            fg=self.EXTRUDE_COLOR,
            bg=self.PANEL_BG,
        )
        self.reward_label.pack(pady=(6, 2))

        self.policy_label = tk.Label(
            right,
            text="Policy: --",
            font=("Courier", 10),
            fg=self.TEXT_DIM,
            bg=self.PANEL_BG,
        )
        self.policy_label.pack(pady=(0, 6))

        # -- Fault indicator --
        self.fault_label = tk.Label(
            right,
            text="",
            font=("Courier", 9, "bold"),
            fg=self.QUALITY_BAD,
            bg=self.PANEL_BG,
        )
        self.fault_label.pack(pady=(0, 4))

        # ---------- Stats bar -----------------------------------------
        stats_frame = tk.Frame(self.root, bg=self.ACCENT, relief=tk.RIDGE, bd=1)
        stats_frame.pack(fill=tk.X, padx=10, pady=(4, 2))
        self.stats_line1 = tk.Label(
            stats_frame,
            text="Step: 0 / 0   |   Reward: 0.00   |   Speed: 0 mm/s",
            font=("Courier", 10),
            fg=self.TEXT,
            bg=self.ACCENT,
        )
        self.stats_line1.pack(anchor="w", padx=8, pady=(4, 0))
        self.stats_line2 = tk.Label(
            stats_frame,
            text="Layer: 0   |   Energy: 0 J   |   Fan: 0%",
            font=("Courier", 10),
            fg=self.TEXT,
            bg=self.ACCENT,
        )
        self.stats_line2.pack(anchor="w", padx=8, pady=(0, 4))

        # ---------- Progress bar --------------------------------------
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor="#0a0a1a",
            background=self.EXTRUDE_COLOR,
            thickness=10,
        )
        self.progress = ttk.Progressbar(
            self.root,
            style="green.Horizontal.TProgressbar",
            length=400,
            mode="determinate",
        )
        self.progress.pack(fill=tk.X, padx=10, pady=(2, 4))

        # ---------- Button bar ----------------------------------------
        btn_frame = tk.Frame(self.root, bg=self.BG)
        btn_frame.pack(pady=(2, 10))

        self.sac_btn = self._make_button(btn_frame, "\u25b6  Run SAC Agent", lambda: self._start("sac"), bold=True)
        self.rand_btn = self._make_button(btn_frame, "\u25b6  Run Random", lambda: self._start("random"))
        self.zero_btn = self._make_button(btn_frame, "\u25b6  Run Zero", lambda: self._start("zero"))
        self.reset_btn = self._make_button(btn_frame, "\u27f3  Reset", self._reset, bg="#444444")

    # ------------------------------------------------------------------ #
    #  Widget helpers                                                     #
    # ------------------------------------------------------------------ #

    def _section_label(self, parent: tk.Frame, text: str) -> None:
        tk.Label(
            parent, text=text, font=("Helvetica", 11, "bold"),
            fg=self.TEXT, bg=self.PANEL_BG,
        ).pack(pady=(10, 4))

    def _separator(self, parent: tk.Frame) -> None:
        sep = tk.Frame(parent, bg=self.ACCENT, height=2)
        sep.pack(fill=tk.X, padx=12, pady=8)

    def _make_button(
        self, parent: tk.Frame, text: str, command, bold: bool = False, bg: str | None = None,
    ) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Helvetica", 10, "bold" if bold else ""),
            bg=bg or self.ACCENT,
            fg="white",
            activebackground=self.BTN_ACTIVE,
            activeforeground="white",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
        )
        btn.pack(side=tk.LEFT, padx=4)
        return btn

    def _make_gauge(
        self, parent: tk.Frame, label_text: str, color: str,
    ) -> dict:
        """Create a horizontal bar gauge with a label and value readout.

        Returns a dict with keys ``bar``, ``label``, ``color`` so we can
        update it later.
        """
        frame = tk.Frame(parent, bg=self.PANEL_BG)
        frame.pack(fill=tk.X, padx=14, pady=2)

        tk.Label(
            frame, text=label_text, font=("Courier", 9),
            fg=self.TEXT, bg=self.PANEL_BG, width=10, anchor="w",
        ).pack(side=tk.LEFT)

        bar = tk.Canvas(frame, height=14, bg="#0a0a1a", highlightthickness=0)
        bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))

        val_label = tk.Label(
            frame, text="--", font=("Courier", 9, "bold"),
            fg=color, bg=self.PANEL_BG, width=9, anchor="e",
        )
        val_label.pack(side=tk.RIGHT)

        return {"bar": bar, "label": val_label, "color": color}

    # ------------------------------------------------------------------ #
    #  Gauge update                                                       #
    # ------------------------------------------------------------------ #

    def _update_gauge(
        self,
        gauge: dict,
        value: float,
        max_val: float,
        fmt: str = "{:.1f}",
    ) -> None:
        bar: tk.Canvas = gauge["bar"]
        label: tk.Label = gauge["label"]
        color: str = gauge["color"]

        bar.delete("all")
        w = bar.winfo_width()
        h = bar.winfo_height()
        if w > 1 and max_val > 0:
            frac = max(0.0, min(value / max_val, 1.0))
            fill_w = frac * w
            bar.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
            # Thin border for readability
            bar.create_rectangle(0, 0, w - 1, h - 1, outline="#22334455")
        label.config(text=fmt.format(value))

    # ================================================================== #
    #  Simulation control                                                 #
    # ================================================================== #

    def _start(self, policy: str) -> None:
        if self.running:
            return
        self._reset()  # clean slate
        self.policy = policy
        self.running = True
        self._set_buttons_enabled(False)
        self.policy_label.config(text=f"Policy: {policy.upper()}")
        thread = threading.Thread(target=self._run_episode, daemon=True)
        thread.start()

    def _reset(self) -> None:
        self.running = False
        self.path_segments = []
        self.step_count = 0
        self.total_steps = 0
        self.total_reward = 0.0
        self.canvas.delete("all")
        self.progress["value"] = 0
        self.reward_label.config(text="Reward: 0.00")
        self.policy_label.config(text="Policy: --")
        self.fault_label.config(text="")
        self.stats_line1.config(text="Step: 0 / 0   |   Reward: 0.00   |   Speed: 0 mm/s")
        self.stats_line2.config(text="Layer: 0   |   Energy: 0 J   |   Fan: 0%")
        self._update_gauge(self.hotend_gauge, 0, 300, "{:.0f}\u00b0C")
        self._update_gauge(self.bed_gauge, 0, 120, "{:.0f}\u00b0C")
        self._update_gauge(self.chamber_gauge, 0, 80, "{:.0f}\u00b0C")
        self._update_gauge(self.adhesion_gauge, 0, 1, "{:.3f}")
        self._update_gauge(self.warping_gauge, 0, 10, "{:.2f} mm")
        self._update_gauge(self.stringing_gauge, 0, 5, "{:.2f} mm")
        self._update_gauge(self.dimerr_gauge, 0, 100, "{:.1f}%")
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self.sac_btn, self.rand_btn, self.zero_btn):
            btn.config(state=state)

    def _on_close(self) -> None:
        self._closing = True
        self.running = False
        self.root.after(100, self.root.destroy)

    # ================================================================== #
    #  Episode runner (background thread)                                 #
    # ================================================================== #

    def _run_episode(self) -> None:
        """Run one full episode in a background thread."""
        try:
            self._run_episode_inner()
        except Exception as exc:
            # Surface errors on the GUI so they are not silently swallowed.
            if not self._closing:
                self.root.after(0, lambda e=exc: self.fault_label.config(text=f"Error: {e}"))
        finally:
            self.running = False
            if not self._closing:
                self.root.after(0, lambda: self._set_buttons_enabled(True))

    def _run_episode_inner(self) -> None:
        from digiprinter.envs.single_agent import PrusaCoreOneEnv
        from digiprinter.envs.wrappers import ClipAction

        env = PrusaCoreOneEnv()
        env = ClipAction(env)
        base_env: PrusaCoreOneEnv = env.unwrapped  # type: ignore[assignment]

        # ----- Load policy model (if SAC) -----------------------------
        model = None
        if self.policy == "sac":
            model = self._try_load_sac_model()
            if model is None:
                # Inform user and fall back to random
                self.root.after(
                    0,
                    lambda: self.fault_label.config(
                        text="SAC model not found \u2014 falling back to random"
                    ),
                )
                self.policy = "random"
                self.root.after(0, lambda: self.policy_label.config(text="Policy: RANDOM (fallback)"))

        # ----- Reset environment --------------------------------------
        obs, info = env.reset(seed=42)
        self.total_reward = 0.0
        total_actions = info.get("total_actions", 0)
        self.total_steps = total_actions if total_actions > 0 else 500
        prev_x: float = base_env.engine.state.x
        prev_y: float = base_env.engine.state.y

        # ----- Episode loop -------------------------------------------
        for step in range(50_000):
            if not self.running or self._closing:
                break

            # Select action
            if self.policy == "sac" and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            elif self.policy == "random":
                action = env.action_space.sample()
            else:  # zero
                action = np.zeros(6, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            self.total_reward += float(reward)
            self.step_count = step + 1

            # Current position
            state = base_env.engine.state
            cx, cy = state.x, state.y
            is_ext = state.is_printing

            self.path_segments.append((prev_x, prev_y, cx, cy, is_ext))
            prev_x, prev_y = cx, cy

            # Schedule UI refresh on the main thread
            if not self._closing:
                self.root.after(0, self._refresh_ui, state, info, step)

            if terminated or truncated:
                break

            # Small delay so the user can see the animation.
            time.sleep(0.02)

        env.close()

    # ------------------------------------------------------------------ #
    #  SAC model loading                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _try_load_sac_model():
        """Attempt to load a trained SAC model. Returns the model or None."""
        try:
            from stable_baselines3 import SAC
        except ImportError:
            return None

        candidates = [
            _PROJECT_ROOT / "checkpoints" / "sac_v2" / "best_model.zip",
            _PROJECT_ROOT / "checkpoints" / "sac_prusa" / "best_model.zip",
            _PROJECT_ROOT / "checkpoints" / "sac_prusa" / "sac_final.zip",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return SAC.load(str(path))
                except Exception:
                    continue
        return None

    # ================================================================== #
    #  UI refresh (main thread)                                           #
    # ================================================================== #

    def _refresh_ui(self, state, info: dict, step: int) -> None:
        """Redraw all gauges, stats, and the latest toolpath segment."""
        if self._closing:
            return

        # -- Temperatures --
        self._update_gauge(self.hotend_gauge, state.hotend_temp, 300, "{:.0f}\u00b0C")
        self._update_gauge(self.bed_gauge, state.bed_temp, 120, "{:.0f}\u00b0C")
        self._update_gauge(self.chamber_gauge, state.chamber_temp, 80, "{:.0f}\u00b0C")

        # -- Quality metrics --
        adhesion = info.get("adhesion", state.adhesion_quality)
        warping = info.get("warping", state.warping_amount)
        stringing = info.get("stringing", state.stringing_amount)
        dim_err = info.get("dimensional_error", state.dimensional_error)

        self._update_gauge(self.adhesion_gauge, adhesion, 1.0, "{:.3f}")
        self._update_gauge(self.warping_gauge, warping, 10.0, "{:.2f} mm")
        self._update_gauge(self.stringing_gauge, stringing, 5.0, "{:.2f} mm")
        self._update_gauge(self.dimerr_gauge, dim_err * 100.0, 100.0, "{:.1f}%")

        # -- Reward --
        color = self.EXTRUDE_COLOR if self.total_reward >= 0 else self.QUALITY_BAD
        self.reward_label.config(text=f"Reward: {self.total_reward:.2f}", fg=color)

        # -- Fault --
        fault = info.get("fault", state.fault)
        if fault:
            self.fault_label.config(text=f"FAULT: {fault}")

        # -- Stats lines --
        self.stats_line1.config(
            text=(
                f"Step: {step + 1} / {self.total_steps}   |   "
                f"Reward: {self.total_reward:.2f}   |   "
                f"Speed: {state.current_speed:.0f} mm/s"
            )
        )
        self.stats_line2.config(
            text=(
                f"Layer: {state.current_layer}/{state.total_layers}   |   "
                f"Energy: {state.total_energy_j:.0f} J   |   "
                f"Fan: {state.fan_speed * 100:.0f}%"
            )
        )

        # -- Progress bar --
        if self.total_steps > 0:
            self.progress["value"] = (step + 1) / self.total_steps * 100.0

        # -- Draw latest toolpath segment on canvas --
        self._draw_latest_segment()

    # ------------------------------------------------------------------ #
    #  Toolpath drawing                                                   #
    # ------------------------------------------------------------------ #

    def _draw_latest_segment(self) -> None:
        """Draw the most recent path segment on the canvas."""
        if not self.path_segments:
            return

        px, py, cx, cy, is_ext = self.path_segments[-1]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        # Compute scale + offset to center the bed in the canvas with a
        # small margin.
        margin = 12
        usable_w = cw - 2 * margin
        usable_h = ch - 2 * margin
        scale = min(usable_w / self.BED_X, usable_h / self.BED_Y)
        ox = (cw - self.BED_X * scale) / 2.0
        oy = (ch - self.BED_Y * scale) / 2.0

        # Map printer mm -> canvas px  (Y flipped so 0 is at bottom)
        x1 = ox + px * scale
        y1 = oy + (self.BED_Y - py) * scale
        x2 = ox + cx * scale
        y2 = oy + (self.BED_Y - cy) * scale

        color = self.EXTRUDE_COLOR if is_ext else self.TRAVEL_COLOR
        width = 2 if is_ext else 1
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def _redraw_bed_outline(self) -> None:
        """Draw a faint outline of the print bed (called once after canvas maps)."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        margin = 12
        usable_w = cw - 2 * margin
        usable_h = ch - 2 * margin
        scale = min(usable_w / self.BED_X, usable_h / self.BED_Y)
        ox = (cw - self.BED_X * scale) / 2.0
        oy = (ch - self.BED_Y * scale) / 2.0
        self.canvas.create_rectangle(
            ox, oy, ox + self.BED_X * scale, oy + self.BED_Y * scale,
            outline="#1a2a3a", width=1, dash=(4, 4),
        )
        # Origin marker
        origin_x = ox
        origin_y = oy + self.BED_Y * scale
        r = 3
        self.canvas.create_oval(
            origin_x - r, origin_y - r, origin_x + r, origin_y + r,
            fill="#334455", outline="",
        )


# =========================================================================== #
#  Entry point                                                                 #
# =========================================================================== #

def main() -> None:
    root = tk.Tk()
    app = DigiPrinterGUI(root)

    # After the window is mapped, draw the bed outline on the canvas.
    def _on_map(event=None):
        app._redraw_bed_outline()

    root.after(200, _on_map)
    root.mainloop()


if __name__ == "__main__":
    main()
