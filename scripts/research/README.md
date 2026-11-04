# Research scripts

These are offline analysis tools used during the study. None of them are needed to run the pipeline or the demo. Run them from the repository root with the package installed (`pip install -e .`).

| Script | What it does | Calls models? |
|---|---|---|
| `ablations.py` | Decomposition controls: full debate vs Phase-1 only vs one model given all the data vs a sham debate | Yes (mock with `STRICT_MODEL=0`) |
| `run_portfolio_batch.py` | Runs `agentic_bl.backtest` for a list of portfolios one after another, with `--resume` | Yes |
| `learn_calibration.py` | Fits q/confidence calibration constants from completed backtests and writes the file `VIEW_CALIBRATION_PATH` points at | No |
| `reconstruct_saved_backtests.py` | Rebuilds portfolio weights from saved views using a new calibration file | No |
| `replay_fixed_scaling.py` | Replays saved views through BL with one fixed, architecture-neutral scaling, and exports stock-month, portfolio-month and portfolio tables | No |
| `run_signal_semantics_suite.py` | Runs `replay_fixed_scaling.py` under three pre-declared signal-translation specs for a robustness appendix | No |

Each script documents its arguments under `--help`.
