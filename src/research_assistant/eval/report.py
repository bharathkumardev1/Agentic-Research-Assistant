"""Where a result came from, and results/RESULTS.md written from the results files."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Optional


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def meta(command: str, settings: Any, embeddings: Any, data_dir: Path) -> Dict[str, Any]:
    tarballs = data_dir / "tarballs.sha256.json"
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        # Dirty means code or configs differ from the commit; results/ is where output goes.
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")),
        "command": command, "date": time.strftime("%Y-%m-%d"), "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "onnxruntime", "langgraph", "openai")},
        "answer_model": settings.answer_model, "review_model": settings.review_model,
        "judge_model": settings.judge_model, "reasoning_effort": settings.reasoning_effort,
        "top_k": settings.top_k, "deadline_seconds": settings.run_deadline_seconds,
        "embeddings": embeddings.name, "cpus": os.cpu_count(),
        "qasper_tarballs_sha256": json.loads(tarballs.read_text()) if tarballs.exists() else None,
    }


def _ci(ci: list) -> str:
    return f"{ci[0]:.3f} to {ci[1]:.3f}"


def section(res: Dict[str, Any]) -> str:
    m, s = res["meta"], res["summary"]
    lines = [
        f"## QASPER {res['split']}: {res['questions']} questions (seed {res['seed']})\n",
        f"`{m['command']}` at commit `{(m['git_sha'] or 'unknown')[:7]}`"
        f"{' with uncommitted changes' if m['git_dirty'] else ''} on {m['date']}. Answers and reviews by "
        f"`{m['answer_model']}`, grading by `{m['judge_model']}` (reasoning effort {m['reasoning_effort']}), "
        f"top {m['top_k']} passages per search, {m['embeddings']} embeddings. Recorded spend "
        f"${res['recorded_spend_usd']:.2f} of a ${res['cap_usd']:.0f} cap"
        + (f"; stopped early: {res['stopped']}" if res.get("stopped") else "") + ".\n",
        "| System | Answer F1 (95% CI) | Evidence F1 (95% CI) | Evidence P / R | Judged correct (95% CI) "
        "| $ per question | p50 / p95 seconds | Calls per question |\n|---|---|---|---|---|---|---|---|",
    ]
    for name, v in s["systems"].items():
        lines.append(f"| {name} | {v['answer_f1']:.3f} ({_ci(v['answer_f1_ci95'])}) | {v['evidence_f1']:.3f} "
                     f"({_ci(v['evidence_f1_ci95'])}) | {v['evidence_precision']:.2f} / {v['evidence_recall']:.2f} | "
                     f"{v['judge_accuracy']:.3f} ({_ci(v['judge_accuracy_ci95'])}) | {v['usd_per_question']:.4f} | "
                     f"{v['seconds_p50']:.1f} / {v['seconds_p95']:.1f} | {v['calls_per_question']:.1f} |")
    if s["loop_minus_single"]:
        lines += ["", "Loop minus single pass, paired by question (10,000 bootstrap resamples):\n",
                  "| Metric | Difference | 95% CI | Loop better | Single pass better |\n|---|---|---|---|---|"]
        lines += [f"| {k} | {v['diff']:+.3f} | {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f} | {v['loop_better']} | "
                  f"{v['single_better']} |" for k, v in s["loop_minus_single"].items()]
    lines += ["", "Where answers went wrong (first failing stage):\n",
              "| System | " + " | ".join(sorted({f for v in s["systems"].values() for f in v["failures"]})) + " |"]
    kinds = sorted({f for v in s["systems"].values() for f in v["failures"]})
    lines.append("|---|" + "---|" * len(kinds))
    lines += [f"| {name} | " + " | ".join(str(v["failures"].get(k, 0)) for k in kinds) + " |"
              for name, v in s["systems"].items()]
    if res.get("judge_check", {}).get("labeled"):
        j = res["judge_check"]
        lines += ["", f"The judge agreed with a person on {j['agreement']:.0%} of {j['labeled']} hand-graded "
                      f"answers (Cohen's kappa {j['cohens_kappa']:.2f}); it passed {j['judge_lenient']} the person "
                      f"failed and failed {j['judge_strict']} the person passed."]
    return "\n".join(lines) + "\n"


def write_markdown(results_dir: Path, path: Path) -> None:
    parts = ["# Results\n"]
    for split in ("dev", "test"):
        f = Path(results_dir) / f"qasper-{split}.json"
        if f.exists():
            parts.append(section(json.loads(f.read_text())))
    Path(path).write_text("\n".join(parts), encoding="utf8")
