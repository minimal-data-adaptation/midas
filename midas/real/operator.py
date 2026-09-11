"""Non-blocking operator labeling for real rollouts."""

from __future__ import annotations

import contextlib
import select
import sys
import termios
import time
import tty


KEY_STATUS = {"1": "success", "0": "failure", "q": "aborted", "2": "aborted", "r": "retry"}


@contextlib.contextmanager
def cbreak_stdin():
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def poll_label() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    return KEY_STATUS.get(sys.stdin.read(1).lower())


def read_final_label(timeout_seconds: float = 300.0, default: str = "failure") -> str:
    if not sys.stdin.isatty():
        return default
    print("Label rollout: [1] success, [0] failure, [q/2] abort, [r] retry")
    deadline = time.monotonic() + timeout_seconds
    with cbreak_stdin():
        while time.monotonic() < deadline:
            status = poll_label()
            if status is not None:
                return status
            time.sleep(0.05)
    return default


__all__ = ["KEY_STATUS", "cbreak_stdin", "poll_label", "read_final_label"]
