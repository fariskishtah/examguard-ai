"""Non-blocking notifications: in-app via queue; optional native macOS banner."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Alert:
    incident_db_id: int
    session_id: int
    track_id: int
    event_type: str
    timestamp_s: float
    confidence: float
    message: str
    screenshot_path: str | None


def native_notify(title: str, message: str, enabled: bool = True, sound: bool = False) -> bool:
    """Fire-and-forget macOS notification; never blocks and never raises."""
    if not enabled or sys.platform != "darwin":
        return False
    safe = lambda s: s.replace("\\", " ").replace('"', "'")[:200]  # noqa: E731
    script = f'display notification "{safe(message)}" with title "{safe(title)}"'
    if sound:
        script += ' sound name "Glass"'
    try:
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False
