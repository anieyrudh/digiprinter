"""G-code tokenizer and parser.

Parses raw G-code text into structured GCodeCommand objects for
downstream interpretation by the simulation engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GCodeCommand:
    """A single parsed G-code command."""

    line_number: int
    command: str  # e.g. "G1", "M104"
    params: dict[str, float]  # e.g. {"X": 10.0, "F": 1200}
    comment: str = ""
    raw: str = ""


# Regex patterns
_COMMAND_RE = re.compile(r"^([GM]\d+)", re.IGNORECASE)
_PARAM_RE = re.compile(r"([A-Z])(-?\d+\.?\d*)", re.IGNORECASE)


class GCodeParser:
    """Stateless parser that converts raw G-code text into GCodeCommand objects."""

    def parse_line(self, line: str, line_number: int = 0) -> GCodeCommand | None:
        """Parse a single line of G-code.

        Parameters
        ----------
        line:
            Raw G-code line, possibly including comments.
        line_number:
            The source line number (1-indexed by convention).

        Returns
        -------
        GCodeCommand or None if the line is empty or comment-only.
        """
        raw = line

        # Strip comment (everything after ';')
        comment = ""
        if ";" in line:
            code_part, comment = line.split(";", 1)
            comment = comment.strip()
        else:
            code_part = line

        code_part = code_part.strip()

        # Skip empty lines
        if not code_part:
            return None

        # Match command (G0, G1, G28, M104, ...)
        cmd_match = _COMMAND_RE.match(code_part)
        if cmd_match is None:
            return None

        command = cmd_match.group(1).upper()

        # Parse parameters from the remainder of the line (after the command)
        remainder = code_part[cmd_match.end():]
        params: dict[str, float] = {}
        for param_match in _PARAM_RE.finditer(remainder):
            letter = param_match.group(1).upper()
            value = float(param_match.group(2))
            params[letter] = value

        return GCodeCommand(
            line_number=line_number,
            command=command,
            params=params,
            comment=comment,
            raw=raw,
        )

    def parse_file(self, gcode_text: str) -> list[GCodeCommand]:
        """Parse a complete G-code program.

        Parameters
        ----------
        gcode_text:
            Multi-line string containing the full G-code program.

        Returns
        -------
        List of GCodeCommand objects (empty/comment-only lines filtered out).
        """
        commands: list[GCodeCommand] = []
        for idx, line in enumerate(gcode_text.splitlines(), start=1):
            cmd = self.parse_line(line, line_number=idx)
            if cmd is not None:
                commands.append(cmd)
        return commands
