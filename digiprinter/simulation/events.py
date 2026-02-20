"""Fault injection and event management for the simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from digiprinter.printer.state import PrinterState


@dataclass
class FaultEvent:
    """A single injectable fault event."""

    fault_type: str  # "thermal_runaway", "clog", "filament_runout", "layer_shift"
    trigger_time: float  # simulation time to trigger (-1 for random)
    duration: float  # how long the fault lasts (0 for permanent)
    severity: float  # 0-1
    active: bool = False


class EventManager:
    """Manages fault injection during a simulation run.

    Faults can be scheduled at specific times or assigned random trigger
    times.  Each simulation tick, ``check_events`` should be called to
    activate / deactivate faults based on the current simulation clock.
    """

    _FAULT_TYPES: list[str] = [
        "thermal_runaway",
        "clog",
        "filament_runout",
        "layer_shift",
    ]

    def __init__(self, rng_seed: int | None = None) -> None:
        self._rng = np.random.default_rng(rng_seed)
        self.events: list[FaultEvent] = []
        self.active_faults: list[FaultEvent] = []

    # ------------------------------------------------------------------ #
    #  Event creation                                                     #
    # ------------------------------------------------------------------ #

    def add_event(
        self,
        fault_type: str,
        trigger_time: float = -1.0,
        duration: float = 0.0,
        severity: float = 1.0,
    ) -> None:
        """Schedule a fault event.

        Parameters
        ----------
        fault_type : str
            One of ``"thermal_runaway"``, ``"clog"``, ``"filament_runout"``,
            ``"layer_shift"``.
        trigger_time : float
            Simulation time (s) at which the fault fires.  A negative value
            causes a random time in [10, 300] s to be assigned.
        duration : float
            Duration of the fault in seconds.  ``0`` means permanent.
        severity : float
            Severity multiplier in [0, 1].
        """
        if trigger_time < 0.0:
            trigger_time = float(self._rng.uniform(10.0, 300.0))

        event = FaultEvent(
            fault_type=fault_type,
            trigger_time=trigger_time,
            duration=duration,
            severity=severity,
        )
        self.events.append(event)

    def add_random_faults(
        self,
        num_faults: int = 1,
        fault_types: list[str] | None = None,
    ) -> None:
        """Generate and schedule *num_faults* random fault events.

        Parameters
        ----------
        num_faults : int
            Number of faults to create.
        fault_types : list[str] | None
            Allowed fault types.  Defaults to all four built-in types.
        """
        if fault_types is None:
            fault_types = self._FAULT_TYPES

        for _ in range(num_faults):
            ft = str(self._rng.choice(fault_types))
            trigger_time = float(self._rng.uniform(10.0, 300.0))
            severity = float(self._rng.uniform(0.3, 1.0))
            # 50 % chance of permanent, otherwise 1-30 s
            if self._rng.random() < 0.5:
                duration = 0.0
            else:
                duration = float(self._rng.uniform(1.0, 30.0))
            self.add_event(ft, trigger_time, duration, severity)

    # ------------------------------------------------------------------ #
    #  Tick                                                               #
    # ------------------------------------------------------------------ #

    def check_events(self, sim_time: float) -> list[FaultEvent]:
        """Check all events against the current simulation clock.

        Activates events whose trigger time has been reached and
        deactivates events whose duration has expired.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.

        Returns
        -------
        list[FaultEvent]
            Newly activated faults (activated during *this* call).
        """
        newly_activated: list[FaultEvent] = []

        for event in self.events:
            # Activate if due
            if not event.active and sim_time >= event.trigger_time:
                event.active = True
                newly_activated.append(event)

            # Deactivate if duration expired (0 = permanent)
            if (
                event.active
                and event.duration > 0.0
                and sim_time >= event.trigger_time + event.duration
            ):
                event.active = False

        # Rebuild active list
        self.active_faults = [e for e in self.events if e.active]

        return newly_activated

    # ------------------------------------------------------------------ #
    #  Fault application                                                  #
    # ------------------------------------------------------------------ #

    def apply_fault(self, fault: FaultEvent, state: PrinterState) -> None:
        """Mutate *state* according to the given fault.

        Parameters
        ----------
        fault : FaultEvent
            The fault to apply.
        state : PrinterState
            Mutable simulation state that will be modified in-place.
        """
        if fault.fault_type == "thermal_runaway":
            state.hotend_temp += 5.0 * fault.severity
            state.fault = "thermal_runaway"

        elif fault.fault_type == "clog":
            state.flow_rate *= 1.0 - 0.8 * fault.severity
            state.fault = "clog"

        elif fault.fault_type == "filament_runout":
            state.flow_rate = 0.0
            state.mass_flow_rate = 0.0
            state.fault = "filament_runout"

        elif fault.fault_type == "layer_shift":
            state.x += float(self._rng.uniform(-2.0, 2.0)) * fault.severity
            state.y += float(self._rng.uniform(-2.0, 2.0)) * fault.severity
            state.fault = "layer_shift"

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all events and active faults."""
        self.events.clear()
        self.active_faults.clear()
