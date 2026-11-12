#!/usr/bin/env python3
"""Reproduce the Magentic Marketplace paper experiments across several LLM providers.

Built on Microsoft's open-source environment (MIT licence):
  https://github.com/microsoft/multi-agent-marketplace
  Bansal et al., "Magentic Marketplace: An Open-Source Environment for
  Studying Agentic Markets", arXiv:2510.25779

What this script adds on top of upstream:
  * one entry point for all four paper experiments
  * one model registry (repro/models.yaml) covering OpenAI, Anthropic,
    Gemini and self-hosted vLLM models
  * resumable runs (finished runs are skipped)
  * a per-run summary CSV (results/summary.csv) with welfare and outcome metrics

Output databases are written to the same paths and file names that the upstream
analysis scripts in experiments/<name>/ expect, so their analysis and plotting
scripts can be run unchanged on these results.

Usage (from the repository root, with Postgres running and .env filled in):
  python repro/run_repro.py --smoke                       # one cheap run end to end
  python repro/run_repro.py --experiments welfare --models gpt-4o qwen3-14b --runs 5
  python repro/run_repro.py --experiments all --models gpt-4.1 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = REPO_ROOT / "repro" / "models.yaml"
SUMMARY_CSV = REPO_ROOT / "results" / "summary.csv"

# Make the upstream package importable even if its editable install isn't picked
# up (seen on macOS with Anaconda Python, where the venv's .pth file can be skipped).
_UPSTREAM_SRC = REPO_ROOT / "packages" / "magentic-marketplace" / "src"
if _UPSTREAM_SRC.is_dir() and str(_UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_SRC))

SUMMARY_FIELDS = [
    "timestamp",
    "experiment",
    "condition",
    "dataset",
    "search_algorithm",
    "search_bandwidth",
    "model",
    "provider",
    "run",
    "customer_welfare",
    "customers",
    "businesses",
    "customers_who_made_purchases",
    "customers_with_needs_met",
    "purchase_completion_rate",
    "total_llm_calls",
    "failed_llm_calls",
    "duration_s",
    "db_path",
    "upstream_commit",
]


# --------------------------------------------------------------------------- #
# Experiment definitions (mirroring the upstream runners in experiments/*)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Condition:
    """One cell of an experiment grid: a dataset plus search settings."""

    name: str
    dataset: str  # relative to repo root
    search_algorithm: str
    search_bandwidth: int


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    export_dir: str
    conditions: tuple[Condition, ...]

    def db_filename(self, cond: Condition, clean_model: str, run: int) -> str:
        # File names follow the upstream runners so their analyze/plot scripts work.
        if self.name == "manipulation":
            return f"malicious_{cond.name}_{clean_model}_run{run}.db"
        if self.name == "position":
            return f"position_{cond.name}_{clean_model}_run{run}.db"
        return f"{self.name}_{cond.name}_{clean_model}_run{run}.db"


MALICIOUS_FOLDERS = [
    f"{domain}_{tactic}"
    for domain in ("contractors", "mexican")
    for tactic in (
        "control",
        "authority",
        "social_proof",
        "loss_aversion",
        "prompt_injection_basic",
        "prompt_injection_strong",
    )
]

POSITION_FOLDERS = [
    "business_0001_first",
    "business_0001_second",
    "business_0001_third",
    "contractors_first",
    "contractors_second",
    "contractors_third",
]

EXPERIMENTS: dict[str, Experiment] = {
    "welfare": Experiment(
        name="welfare",
        description="Customer welfare under realistic (lexical) vs perfect (optimal) search",
        export_dir="paper_experiments/welfare/results",
        conditions=tuple(
            Condition(f"{label}_{algo}", path, algo, 10)
            for label, path in (
                ("mexican_33_99", "data/mexican_33_99"),
                ("contractors_33_99", "data/contractors/contractors_33_99"),
            )
            for algo in ("lexical", "optimal")
        ),
    ),
    "search": Experiment(
        name="search",
        description="Consideration-set size (search results shown) vs welfare",
        export_dir="paper_experiments/consideration_set_size/results",
        conditions=tuple(
            Condition(f"mexican_100_300_limit_{k}", "data/mexican_100_300", "lexical", k)
            for k in (3, 10, 30, 100)
        ),
    ),
    "manipulation": Experiment(
        name="manipulation",
        description="Resistance to manipulative seller descriptions and prompt injection",
        export_dir="paper_experiments/malicious/results",
        conditions=tuple(
            Condition(f, f"data/malicious_description/{f}", "lexical", 10)
            for f in MALICIOUS_FOLDERS
        ),
    ),
    "position": Experiment(
        name="position",
        description="Search-position and first-proposal bias",
        export_dir="paper_experiments/position/results",
        conditions=tuple(
            Condition(f, f"data/position_bias/{f}", "simple", 10) for f in POSITION_FOLDERS
        ),
    ),
}

SMOKE = Experiment(
    name="smoke",
    description="One small run to check the whole pipeline (3 customers, 9 businesses)",
    export_dir="paper_experiments/smoke/results",
    conditions=(Condition("mexican_3_9_lexical", "data/mexican_3_9", "lexical", 10),),
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_models() -> dict[str, dict]:
    with MODELS_FILE.open() as f:
        raw = yaml.safe_load(f)["models"]
    models = {}
    for key, cfg in raw.items():
        models[key] = {k: os.path.expandvars(str(v)) for k, v in cfg.items()}
    return models


def clean_model_name(model: str) -> str:
    """Same normalisation as the upstream runners."""
    return model.replace("-", "_").replace(".", "_").replace("/", "_")


def schema_name(*parts: str) -> str:
    """Postgres schema names are limited to 63 chars; keep them unique with a hash."""
    name = "_".join(parts).lower().replace("-", "_").replace(".", "_").replace("/", "_")
    if len(name) <= 63:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{name[:54]}_{digest}"


def upstream_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class ModelEnv:
    """Set the environment variables the upstream LLM clients read, then restore them."""

    KEYS = ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MAX_CONCURRENCY")

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.KEYS}
        os.environ["LLM_PROVIDER"] = self.cfg["provider"]
        os.environ["LLM_MODEL"] = self.cfg["model"]
        if "base_url" in self.cfg:
            os.environ["OPENAI_BASE_URL"] = self.cfg["base_url"]
            key = self.cfg.get("api_key", "")
            os.environ["OPENAI_API_KEY"] = key if key and "$" not in key else "dummy"
        else:
            os.environ.pop("OPENAI_BASE_URL", None)
        if self.cfg.get("max_concurrency"):
            os.environ["LLM_MAX_CONCURRENCY"] = self.cfg["max_concurrency"]
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def append_summary(row: dict) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def check_preflight(models: dict[str, dict], selected: list[str]) -> list[str]:
    """Return a list of problems that would make the runs fail."""
    problems = []
    for key in selected:
        cfg = models[key]
        provider = cfg["provider"]
        if "base_url" in cfg:  # self-hosted (vLLM) model
            if not cfg["base_url"] or "$" in cfg["base_url"]:
                problems.append(f"{key}: set VLLM_BASE_URL in .env (e.g. http://localhost:8001/v1)")
            continue
        env_key = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[provider]
        val = os.environ.get(env_key, "")
        if not val or val == "todo":
            problems.append(f"{key}: {env_key} is not set in .env")
    return problems


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


async def run_one(exp: Experiment, cond: Condition, model_key: str, cfg: dict, run: int) -> None:
    from magentic_marketplace.experiments.run_analytics import run_analytics
    from magentic_marketplace.experiments.run_experiment import run_marketplace_experiment

    clean = clean_model_name(model_key)
    export_dir = REPO_ROOT / exp.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    db_file = exp.db_filename(cond, clean, run)
    db_path = export_dir / db_file

    if db_path.exists():
        print(f"  skip  {db_file} (already done)")
        return

    schema = schema_name("r", exp.name[:4], cond.name, clean, f"r{run}")
    print(f"  run   {exp.name} | {cond.name} | {model_key} | run {run}")
    t0 = time.time()

    with ModelEnv(cfg):
        await run_marketplace_experiment(
            data_dir=REPO_ROOT / cond.dataset,
            experiment_name=schema,
            search_algorithm=cond.search_algorithm,
            search_bandwidth=cond.search_bandwidth,
            customer_max_steps=100,
            postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
            postgres_password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            override=True,
            export_sqlite=True,
            export_dir=str(export_dir),
            export_filename=db_file,
        )

    results = await run_analytics(
        str(db_path), db_type="sqlite", save_to_json=False, print_results=False
    )

    # Keep the full analytics next to the database for later analysis.
    db_path.with_suffix(".analytics.json").write_text(results.model_dump_json(indent=2))

    # The consideration-set analysis upstream reads JSON files with this naming.
    if exp.name == "search":
        dataset = Path(cond.dataset).name
        upstream_json = export_dir / (
            f"analytics_results_search_limit_{cfg['model'].replace('/', '_')}_{dataset}_"
            f"limit_{cond.search_bandwidth}_run_{run}.json"
        )
        upstream_json.write_text(results.model_dump_json(indent=2))

    append_summary(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment": exp.name,
            "condition": cond.name,
            "dataset": cond.dataset,
            "search_algorithm": cond.search_algorithm,
            "search_bandwidth": cond.search_bandwidth,
            "model": model_key,
            "provider": cfg["provider"],
            "run": run,
            "customer_welfare": results.total_marketplace_customer_utility,
            "customers": results.total_customers,
            "businesses": results.total_businesses,
            "customers_who_made_purchases": results.customers_who_made_purchases,
            "customers_with_needs_met": results.customers_with_needs_met,
            "purchase_completion_rate": results.purchase_completion_rate,
            "total_llm_calls": results.total_llm_calls,
            "failed_llm_calls": results.failed_llm_calls,
            "duration_s": round(time.time() - t0, 1),
            "db_path": str(db_path.relative_to(REPO_ROOT)),
            "upstream_commit": upstream_commit(),
        }
    )
    print(
        f"        welfare={results.total_marketplace_customer_utility:.1f} "
        f"needs_met={results.customers_with_needs_met}/{results.total_customers} "
        f"({time.time() - t0:.0f}s)"
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["welfare"],
        help=f"Which experiments to run: {', '.join(EXPERIMENTS)} or 'all'",
    )
    p.add_argument("--models", nargs="+", default=["gpt-5-nano"], help="Keys from repro/models.yaml")
    p.add_argument("--runs", type=int, default=5, help="Runs per condition (paper uses 5)")
    p.add_argument("--conditions", nargs="+", help="Only run conditions whose name contains one of these")
    p.add_argument("--smoke", action="store_true", help="One small run with the first model, to test setup")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    return p.parse_args(argv)


def build_plan(args, models) -> list[tuple[Experiment, Condition, str, int]]:
    if args.smoke:
        return [(SMOKE, SMOKE.conditions[0], args.models[0], 1)]
    names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        sys.exit(f"Unknown experiment(s): {unknown}. Choose from {list(EXPERIMENTS)} or 'all'.")
    plan = []
    for model in args.models:
        for name in names:
            exp = EXPERIMENTS[name]
            for cond in exp.conditions:
                if args.conditions and not any(c in cond.name for c in args.conditions):
                    continue
                for run in range(1, args.runs + 1):
                    plan.append((exp, cond, model, run))
    return plan


async def main(argv=None) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    models = load_models()
    missing = [m for m in args.models if m not in models]
    if missing:
        sys.exit(f"Unknown model(s): {missing}. Add them to repro/models.yaml. Known: {list(models)}")

    plan = build_plan(args, models)
    print(f"Plan: {len(plan)} runs")
    for exp, cond, model, run in plan[:12]:
        print(f"  {exp.name:12s} {cond.name:40s} {model:20s} run {run}")
    if len(plan) > 12:
        print(f"  ... and {len(plan) - 12} more")

    if args.dry_run:
        return 0

    problems = check_preflight(models, sorted({m for _, _, m, _ in plan}))
    if problems:
        print("\nSetup problems:")
        for p in problems:
            print(f"  - {p}")
        return 1

    failures = 0
    for exp, cond, model, run in plan:
        try:
            await run_one(exp, cond, model, models[model], run)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # keep going; a failed run is retried next time
            failures += 1
            print(f"  FAIL  {exp.name} | {cond.name} | {model} | run {run}: {type(e).__name__}: {e}")
    print(f"\nDone. {len(plan) - failures} ok, {failures} failed. Summary: {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
