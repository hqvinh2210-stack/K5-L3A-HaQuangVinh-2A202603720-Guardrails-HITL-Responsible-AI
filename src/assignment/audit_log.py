"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[str, str, float]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Record the sanitized input and the start time for a request."""
        self._open[(user_id, request_id)] = (
            content_filter(text)["redacted"], utc_now_iso(), time.monotonic()
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Pair the sanitized response with its input and decision."""
        previous = self._open.pop((user_id, request_id), None)
        if previous is None:
            user_input, timestamp, latency_ms = None, utc_now_iso(), None
        else:
            user_input, timestamp, started_at = previous
            latency_ms = round((time.monotonic() - started_at) * 1000, 3)
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "input": user_input,
            "output": content_filter(text)["redacted"],
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
