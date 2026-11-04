"""
Run the diversified portfolios one after another.

This version has stronger --resume logic:
  - It detects new batch files by portfolio_id metadata.
  - It also detects older frontend-style filenames such as:
      backtest_AAMJCH_2024-01-01_2024-12-31_gemma_debate_3ca0de32.json
    where AAMJCH = AAPL,AMZN,MA,JNJ,COP,HON.

Run from project root:
  python scripts/research/run_portfolio_batch.py --mode gemma_debate --provider live --start 2024-01-31 --end 2024-12-31 --resume

Useful:
  --dry-run                 print what would run/skip
  --resume                  skip completed portfolios
  --strict / --no-strict     default strict on
  --continue-on-error        keep going if one portfolio fails
  --calibration-path PATH    set VIEW_CALIBRATION_PATH for holdout runs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from agentic_bl import config
PORTFOLIOS = [
    # {"id": "p01_googl_nvda_jpm_lly_xom_cost", "code": "GNJLXC", "tickers": ["GOOGL", "NVDA", "JPM", "LLY", "XOM", "COST"]},
    {"id": "p02_msft_meta_v_unh_cvx_cat", "code": "MMVUCC", "tickers": ["MSFT", "META", "V", "UNH", "CVX", "CAT"]},
    {"id": "p03_aapl_amzn_ma_jnj_cop_hon", "code": "AAMJCH", "tickers": ["AAPL", "AMZN", "MA", "JNJ", "COP", "HON"]},
    {"id": "p04_avgo_nflx_bac_mrk_slb_wmt", "code": "ANBMSW", "tickers": ["AVGO", "NFLX", "BAC", "MRK", "SLB", "WMT"]},
    {"id": "p05_amd_dis_gs_abbv_lin_hd", "code": "ADGALH", "tickers": ["AMD", "DIS", "GS", "ABBV", "LIN", "HD"]},
    {"id": "p06_orcl_tmus_ms_tmo_fcx_low", "code": "OTMTFL", "tickers": ["ORCL", "TMUS", "MS", "TMO", "FCX", "LOW"]},
    {"id": "p07_crm_tsla_jpm_lly_nee_pg", "code": "CTJLNP", "tickers": ["CRM", "TSLA", "JPM", "LLY", "NEE", "PG"]},
    {"id": "p08_nvda_mcd_v_xom_ba_pld", "code": "NMVXBP", "tickers": ["NVDA", "MCD", "V", "XOM", "BA", "PLD"]},
    {"id": "p09_googl_pep_ma_unh_ge_amt", "code": "GPMUGA", "tickers": ["GOOGL", "PEP", "MA", "UNH", "GE", "AMT"]},
    {"id": "p10_msft_ko_gs_cvx_rtx_eqix", "code": "MKGCRE", "tickers": ["MSFT", "KO", "GS", "CVX", "RTX", "EQIX"]},
]


def _load_json_light(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _json_has_exact_match(
    data: dict[str, Any],
    portfolio_id: str,
    tickers: list[str],
    start: str,
    end: str,
    mode: str,
) -> bool:
    """Check newer JSON metadata and a few common locations."""
    wanted_tickers = set(tickers)
    candidates = [
        data,
        data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        data.get("config", {}) if isinstance(data.get("config"), dict) else {},
    ]

    has_portfolio = False
    has_dates = False
    has_mode = False

    for obj in candidates:
        if not isinstance(obj, dict):
            continue

        if obj.get("portfolio_id") == portfolio_id:
            has_portfolio = True

        universe = obj.get("universe") or obj.get("tickers")
        if isinstance(universe, list) and set(map(str, universe)) == wanted_tickers:
            has_portfolio = True

        if obj.get("start") == start and obj.get("end") == end:
            has_dates = True

        if obj.get("start_date") == start and obj.get("end_date") == end:
            has_dates = True

        if obj.get("run_mode") == mode or obj.get("mode") == mode:
            has_mode = True

    return has_portfolio and has_dates and has_mode


def _filename_has_existing_run(
    path: Path,
    portfolio_code: str,
    start: str,
    end: str,
    mode: str,
    loose_january: bool = True,
) -> bool:
    """Detect older frontend names like backtest_AAMJCH_2024-01-01_2024-12-31_gemma_debate_x.json."""
    name = path.name

    if not name.startswith("backtest_") or not name.endswith(".json"):
        return False
    if portfolio_code not in name:
        return False
    if mode not in name:
        return False
    if end not in name:
        return False
    if start in name:
        return True

    # Your older frontend runs used 2024-01-01 while the batch script often uses 2024-01-31.
    # For a Jan-to-Dec yearly run, treat either Jan 1 or Jan 31 as already covering the same year.
    if loose_january:
        yyyy = start[:4]
        same_year_jan_start = f"{yyyy}-01-01" in name or f"{yyyy}-01-31" in name
        same_year_end = end.startswith(yyyy + "-") or end.startswith(str(int(yyyy) + 1) + "-")
        if same_year_jan_start and same_year_end:
            return True

    return False


def existing_backtest_for(
    portfolio_id: str,
    portfolio_code: str,
    tickers: list[str],
    start: str,
    end: str,
    mode: str,
) -> list[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []

    matches: list[Path] = []

    for path in runs_dir.glob("backtest_*.json"):
        # Fast old-frontend filename check.
        if _filename_has_existing_run(path, portfolio_code, start, end, mode):
            matches.append(path)
            continue

        # Metadata check for newer batch files.
        data = _load_json_light(path)
        if data and _json_has_exact_match(data, portfolio_id, tickers, start, end, mode):
            matches.append(path)
            continue

        # Fallback text check for patch/version differences.
        try:
            txt = path.read_text(errors="ignore")
        except Exception:
            continue

        if (
            f'"portfolio_id": "{portfolio_id}"' in txt
            and f'"start": "{start}"' in txt
            and f'"end": "{end}"' in txt
            and (f'"run_mode": "{mode}"' in txt or f'"mode": "{mode}"' in txt)
        ):
            matches.append(path)

    return sorted(set(matches))


def run_one(args: argparse.Namespace, portfolio: dict[str, Any], index: int, total: int) -> int:
    tickers = portfolio["tickers"]
    universe = ",".join(tickers)
    portfolio_id = f"{args.portfolio_prefix}_{portfolio['id']}"
    portfolio_code = portfolio["code"]

    if args.resume:
        existing = existing_backtest_for(
            portfolio_id=portfolio_id,
            portfolio_code=portfolio_code,
            tickers=tickers,
            start=args.start,
            end=args.end,
            mode=args.mode,
        )
        if existing:
            print(f"\n[{index}/{total}] SKIP existing {portfolio_id} | {universe}", flush=True)
            for p in existing[:3]:
                print(f"  found: {p}", flush=True)
            if len(existing) > 3:
                print(f"  ... and {len(existing) - 3} more", flush=True)
            return 0

    cmd = [
        sys.executable,
        "-m",
        "agentic_bl.backtest",
        "--provider",
        args.provider,
        "--mode",
        args.mode,
        "--start",
        args.start,
        "--end",
        args.end,
        "--portfolio-id",
        portfolio_id,
        "--universe",
        universe,
    ]

    if args.strict:
        cmd.append("--strict")

    env = os.environ.copy()

    # Match the frontend/server environment by default. The previous behavior
    # silently removed VIEW_CALIBRATION_PATH whenever --calibration-path was
    # omitted, so UI and batch runs could use different calibration settings.
    if args.no_calibration:
        env.pop("VIEW_CALIBRATION_PATH", None)
    elif args.calibration_path:
        env["VIEW_CALIBRATION_PATH"] = args.calibration_path
    # else: inherit VIEW_CALIBRATION_PATH from the shell unchanged.

    env.setdefault("BL_PRIOR_MODE", "equal_weight")
    env.setdefault("STRICT_MODEL", "1" if args.strict else "0")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{portfolio_id}_{args.start}_{args.end}_{args.mode}.log"

    print("\n" + "=" * 96, flush=True)
    print(f"[{index}/{total}] Running {portfolio_id}", flush=True)
    print(f"Old filename code: {portfolio_code}", flush=True)
    print(f"Universe: {universe}", flush=True)
    print(f"Mode: {args.mode} | Provider: {args.provider} | Dates: {args.start} -> {args.end}", flush=True)
    print(f"Log: {log_path}", flush=True)
    print("Command:", " ".join(cmd), flush=True)
    print(
        "Effective env: "
        f"VIEW_CALIBRATION_PATH={env.get('VIEW_CALIBRATION_PATH', '')!r} | "
        f"BL_PRIOR_MODE={env.get('BL_PRIOR_MODE', '')!r}",
        flush=True,
    )

    if args.dry_run:
        return 0

    with log_path.open("w") as log:
        log.write("Command: " + " ".join(cmd) + "\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)

        rc = proc.wait()

    if rc == 0:
        print(f"[{index}/{total}] DONE {portfolio_id}", flush=True)
    else:
        print(f"[{index}/{total}] FAILED {portfolio_id} with exit code {rc}", flush=True)

    if args.sleep_seconds > 0:
        time.sleep(args.sleep_seconds)

    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["live", "fixture"], default="live")
    parser.add_argument(
        "--mode",
        choices=[
            "gemma_debate", "gemma_single", "gemini_c1",
            "gemini_debate", "compare", "factorial",
        ],
        default="gemma_debate",
    )
    parser.add_argument("--start", default="2024-01-31")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--portfolio-prefix", default="batch")
    parser.add_argument("--log-dir", default="batch_logs")
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="Override VIEW_CALIBRATION_PATH for child runs. Omit to inherit the shell value.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Explicitly remove VIEW_CALIBRATION_PATH for uncalibrated training runs.",
    )
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Skip portfolios that already have a matching saved backtest JSON.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not Path("src/agentic_bl/backtest.py").exists():
        print("ERROR: run this script from the repository root.", file=sys.stderr)
        return 2

    if args.no_calibration and args.calibration_path:
        print("ERROR: use either --calibration-path or --no-calibration, not both.", file=sys.stderr)
        return 2

    # Fail fast rather than spending hours on model calls and then discovering
    # that the optimizer treated every name as one capped Unknown sector.
    all_batch_tickers = sorted({t for p in PORTFOLIOS for t in p["tickers"]})
    unknown = [t for t in all_batch_tickers if config.sector_of(t) == "Unknown"]
    if unknown:
        print(
            "ERROR: missing sector mappings for batch tickers: " + ",".join(unknown),
            file=sys.stderr,
        )
        print(
            "Add them to config.SECTOR_BY_TICKER before running the batch.",
            file=sys.stderr,
        )
        return 2

    failures = []

    for idx, portfolio in enumerate(PORTFOLIOS, start=1):
        rc = run_one(args, portfolio, idx, len(PORTFOLIOS))
        if rc != 0:
            failures.append((portfolio["id"], rc))
            if not args.continue_on_error:
                print("\nStopping after first failure. Re-run with --continue-on-error to keep going.", file=sys.stderr)
                return rc

    print("\n" + "=" * 96)
    if failures:
        print("Batch finished with failures:")
        for pid, rc in failures:
            print(f"  {pid}: exit code {rc}")
        return 1

    print("Batch finished successfully.")
    print("Backtest JSONs are in: runs/")
    print("Logs are in:", args.log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())