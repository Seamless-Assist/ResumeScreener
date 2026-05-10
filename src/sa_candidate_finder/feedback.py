from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from sa_candidate_finder.config import load_config
from sa_candidate_finder.models import RunTelemetry


def run_feedback(jd_path: Path) -> None:
    cfg = load_config()

    # Find most recent run for this JD
    run = _find_last_run(jd_path, cfg)
    if run is None:
        typer.echo(f"No previous search runs found for: {jd_path}")
        raise typer.Exit(1)

    typer.echo(f"\nProviding feedback for run {run['run_id']} ({run['started_at']})")
    typer.echo(f"JD: {jd_path}")

    # Show candidates from the results file
    results_path = _find_results_file(jd_path, run["run_id"], cfg)
    if results_path:
        typer.echo(f"Results file: {results_path}")

    feedback_entries: list[dict] = []

    while True:
        typer.echo("\nEnter feedback (or 'done' to finish):")
        typer.echo("  Format: <candidate_name_or_id> [+|-] [optional note]")
        typer.echo("  Example: Jane Smith + Strong Python fit")
        typer.echo("  Example: John Doe - Weak communication skills")
        entry = typer.prompt("Feedback").strip()

        if entry.lower() == "done":
            break

        parts = entry.split(maxsplit=2)
        if len(parts) < 2:
            typer.echo("  Invalid format. Use: <name_or_id> [+|-] [note]")
            continue

        # Handle name with signal at end
        signal_char = None
        for i, p in enumerate(parts):
            if p in ("+", "-"):
                signal_char = p
                name_parts = parts[:i]
                note = " ".join(parts[i + 1:])
                break
        else:
            typer.echo("  Missing signal. Use + for selected, - for rejected.")
            continue

        feedback_entries.append({
            "run_id": run["run_id"],
            "jd_hash": run["jd_hash"],
            "candidate_ref": " ".join(name_parts),
            "signal": "positive" if signal_char == "+" else "negative",
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        typer.echo(f"  Recorded: {'Selected' if signal_char == '+' else 'Rejected'} — {' '.join(name_parts)}")

    if feedback_entries:
        _save_feedback(feedback_entries, cfg)
        typer.echo(f"\n{len(feedback_entries)} feedback entries saved.")
        typer.echo("Run 'SACandidateFinder learn' to apply feedback.")
    else:
        typer.echo("\nNo feedback recorded.")


def _find_last_run(jd_path: Path, cfg) -> Optional[dict]:
    log_path = Path(cfg.telemetry_log_path)
    if not log_path.exists():
        return None

    jd_stem = jd_path.stem
    last = None
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if jd_stem in entry.get("jd_hash", "") or True:  # match by hash later
                    last = entry
            except json.JSONDecodeError:
                continue
    return last


def _find_results_file(jd_path: Path, run_id: str, cfg) -> Optional[Path]:
    results_dir = Path(cfg.results_dir)
    if not results_dir.exists():
        return None
    matches = list(results_dir.glob(f"{jd_path.stem}_*.md"))
    return sorted(matches)[-1] if matches else None


def _save_feedback(entries: list[dict], cfg) -> None:
    log_path = Path(cfg.feedback_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
