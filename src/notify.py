import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _format_summary(summary: dict) -> str:
    lines = []

    metrics = summary.get("metrics", {})
    if metrics:
        lines.append("\n── RAPM Validation ──────────────────────────────")
        for key in ["offense", "defense", "total"]:
            rmse = metrics.get(f"rmse_{key}", "n/a")
            corr = metrics.get(f"corr_{key}", "n/a")
            lines.append(f"  {key:8s}  RMSE: {rmse:<8}  Corr: {corr}")
        lines.append(f"  Matched: {metrics.get('n_players_matched', 'n/a')} players")

    top10 = summary.get("top10_combined", [])
    if top10:
        lines.append("\n── Top 10 Combined ──────────────────────────────")
        for i, p in enumerate(top10, 1):
            name = p.get("player_name") or p["player_id"]
            off  = p.get("offensive_rating", "")
            dfn  = p.get("defensive_rating", "")
            tot  = p.get("combined_rating",  "")
            lines.append(f"  {i:2}. {str(name):<26}  O: {off:>7.3f}  D: {dfn:>7.3f}  TOT: {tot:>7.3f}")

    params = summary.get("params", {})
    if params:
        lines.append("\n── Run Parameters ───────────────────────────────")
        for k, v in params.items():
            if v is not None:
                lines.append(f"  {k}: {v}")

    run_id = summary.get("mlflow_run_id")
    if run_id:
        ui = summary.get("mlflow_tracking_uri", "http://localhost:5000")
        lines.append(f"\nMLflow run: {run_id}")
        lines.append(f"View:       {ui}  (run: mlflow ui)")
    else:
        summary_path = _ROOT / "results" / "run_summary.json"
        lines.append(f"\nFull summary: {summary_path}")

    return "\n".join(lines)


def send_completion_email(
    job: dict,
    stages_run: list[str],
    error: Exception | None = None,
    summary: dict | None = None,
) -> None:
    cfg = job.get("email", {})
    if not cfg.get("enabled", False):
        return

    to_addr   = cfg["to"]
    from_addr = cfg.get("from", to_addr)
    smtp_user = os.environ.get("GMAIL_USER", from_addr)
    smtp_pass = os.environ.get("GMAIL_APP_PASSWORD")

    if not smtp_pass:
        print("  Email skipped: GMAIL_APP_PASSWORD environment variable not set")
        return

    status  = "FAILED" if error else "completed"
    subject = f"NBA pipeline {status} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    lines = [f"Stages run: {', '.join(stages_run)}"]
    if error:
        lines.append(f"\nError: {error}")
    else:
        lines.append("\nAll stages completed successfully.")
        if summary:
            lines.append(_format_summary(summary))

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg.set_content("\n".join(lines))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
        print(f"  Notification sent to {to_addr}")
    except Exception as exc:
        print(f"  Email notification failed: {exc}")
