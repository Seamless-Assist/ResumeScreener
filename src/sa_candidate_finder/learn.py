from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer

from sa_candidate_finder.config import load_config


def run_learn() -> None:
    cfg = load_config()
    feedback_path = Path(cfg.feedback_log_path)

    if not feedback_path.exists():
        typer.echo("No feedback log found. Provide feedback first with 'SACandidateFinder feedback'.")
        raise typer.Exit(0)

    entries = _load_feedback(feedback_path)
    if not entries:
        typer.echo("No feedback entries to process.")
        raise typer.Exit(0)

    typer.echo(f"\nProcessing {len(entries)} feedback entries...")

    # Aggregate signals per candidate reference
    signals: dict[str, dict] = defaultdict(lambda: {"positive": 0, "negative": 0, "notes": []})
    for e in entries:
        ref = e.get("candidate_ref", "")
        sig = e.get("signal", "")
        note = e.get("note", "")
        if sig == "positive":
            signals[ref]["positive"] += 1
        elif sig == "negative":
            signals[ref]["negative"] += 1
        if note:
            signals[ref]["notes"].append(note)

    # Save summarised learning profile
    learn_path = Path(cfg.telemetry_log_path).parent / "learning_profile.jsonl"
    learn_path.parent.mkdir(parents=True, exist_ok=True)

    with learn_path.open("w", encoding="utf-8") as f:
        for ref, data in signals.items():
            f.write(json.dumps({"candidate_ref": ref, **data}) + "\n")

    typer.echo(f"Learning profile updated: {learn_path}")
    typer.echo(f"  {len(signals)} unique candidates profiled.")
    typer.echo("  Future searches will use this profile for refined evaluation guidance.")
    typer.echo("\nNote: Learning effects apply on next search run.")


def _load_feedback(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries
